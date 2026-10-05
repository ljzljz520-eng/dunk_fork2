import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.console import ConsoleOptions, RenderResult
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from unidiff import PatchedFile

from dunk.underline_bar import UnderlineBar

# ---------------------------------------------------------------------------
# Untrusted-input handling.
#
# Everything originating from patch metadata (file names, hunk section
# headers, status strings) is treated as untrusted.  Two guarantees:
#   * Rich markup ("[red]...") can never be interpreted -- all such strings
#     are rendered as Text/Segment content, never parsed as markup.
#   * C0/C1 control characters (including ESC, which enables OSC terminal
#     injection) and bidirectional formatting characters are replaced with
#     visible escape sequences.
# ---------------------------------------------------------------------------

# C0 controls with familiar short forms.
_C0_ESCAPES = {
    0x00: r"\0",
    0x07: r"\a",
    0x08: r"\b",
    0x09: r"\t",
    0x0A: r"\n",
    0x0B: r"\v",
    0x0C: r"\f",
    0x0D: r"\r",
    0x1B: r"\e",
}

# Bidirectional formatting characters (Unicode bidi algorithm controls).
_BIDI_CONTROLS = {
    0x061C,  # ALM
    0x200E,  # LRM
    0x200F,  # RLM
    0x202A,  # LRE
    0x202B,  # RLE
    0x202C,  # PDF
    0x202D,  # LRO
    0x202E,  # RLO
    0x2066,  # LRI
    0x2067,  # RLI
    0x2068,  # FSI
    0x2069,  # PDI
}

# All C0 controls + DEL + C1 controls + bidi controls (metadata strings).
_UNSAFE_TEXT_RE = re.compile(
    "[\x00-\x1f\x7f\x80-\x9f"
    "\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]"
)

# Same as above, but tab and newline are structural in code bodies and are
# kept so syntax highlighting / line layout still works.
_UNSAFE_CODE_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f"
    "\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]"
)

ESCAPED_CONTROL_STYLE = "bold yellow"


def _escape_control_char(char: str) -> Optional[str]:
    codepoint = ord(char)
    if codepoint in _BIDI_CONTROLS:
        return f"\\u{codepoint:04X}"
    if codepoint < 0x20:
        return _C0_ESCAPES.get(codepoint, f"\\x{codepoint:02X}")
    if codepoint == 0x7F or 0x80 <= codepoint <= 0x9F:
        return f"\\x{codepoint:02X}"
    return None


def sanitize_text(value: str, style: Optional[str] = None) -> Text:
    """Render an untrusted metadata string as structured Text.

    Rich markup is never interpreted (Text content is literal) and control
    characters are shown as visible escape sequences.
    """
    text = Text(style=style)
    position = 0
    for match in _UNSAFE_TEXT_RE.finditer(value):
        if match.start() > position:
            text.append(value[position : match.start()])
        text.append(
            _escape_control_char(match.group()), style=ESCAPED_CONTROL_STYLE
        )
        position = match.end()
    if position < len(value):
        text.append(value[position:])
    return text


def sanitize_code(value: str) -> str:
    """Neutralise dangerous control characters in a code body.

    Tab and newline are preserved (they are structural); every other C0
    control, DEL, C1 control and bidi control is replaced with a visible
    escape sequence.  Syntax highlighting is unaffected.
    """
    return _UNSAFE_CODE_RE.sub(
        lambda match: _escape_control_char(match.group()), value
    )


# Git quotes "unusual" path names (see git's quote_c_style) using a leading/
# trailing double quote, C-style escapes for control characters and octal
# escapes for high bytes.  unidiff does not decode these.
_GIT_SIMPLE_ESCAPES = {
    "a": 0x07,
    "b": 0x08,
    "t": 0x09,
    "n": 0x0A,
    "v": 0x0B,
    "f": 0x0C,
    "r": 0x0D,
    '"': 0x22,
    "\\": 0x5C,
}


def decode_git_path(raw: str) -> Optional[str]:
    """Decode a path from a diff header using Git path semantics.

    Returns the decoded path, or ``None`` if it is malformed/unsafe.
    """
    if len(raw) < 2 or not (raw.startswith('"') and raw.endswith('"')):
        return raw

    result = bytearray()
    index = 1
    end = len(raw) - 1
    while index < end:
        char = raw[index]
        if char != "\\":
            # Inside a quoted path every raw byte must be ASCII; high bytes
            # are always octal-escaped by Git.
            if ord(char) > 0x7F:
                return None
            result.append(ord(char))
            index += 1
            continue

        index += 1
        if index >= end:
            return None
        escaped = raw[index]
        if escaped in _GIT_SIMPLE_ESCAPES:
            result.append(_GIT_SIMPLE_ESCAPES[escaped])
            index += 1
        elif escaped in "01234567":
            value = 0
            digits = 0
            while digits < 3 and index < end and raw[index] in "01234567":
                value = value * 8 + int(raw[index])
                index += 1
                digits += 1
            result.append(value & 0xFF)
        else:
            # Unknown escape -- Git rejects these, so do we.
            return None

    try:
        return result.decode("utf-8")
    except UnicodeDecodeError:
        return None


def strip_git_prefix(path: str) -> str:
    """Strip the ``a/``/``b/`` VCS prefix Git adds to diff file names."""
    if (
        len(path) >= 2
        and path[0] in "ab"
        and path[1] == "/"
        and not path.startswith(("a//", "b//"))
    ):
        return path[2:]
    return path


def simple_pluralise(word: str, number: int) -> str:
    if number == 1:
        return word
    else:
        return word + "s"


@dataclass
class PatchSetHeader:
    file_modifications: int
    file_additions: int
    file_removals: int
    line_additions: int
    line_removals: int

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.file_modifications:
            yield Align.center(
                Text.assemble(
                    (str(self.file_modifications), "blue"),
                    (
                        f" {simple_pluralise('file', self.file_modifications)} changed",
                        "blue",
                    ),
                )
            )
        if self.file_additions:
            yield Align.center(
                Text.assemble(
                    (str(self.file_additions), "green"),
                    (
                        f" {simple_pluralise('file', self.file_additions)} added",
                        "green",
                    ),
                )
            )
        if self.file_removals:
            yield Align.center(
                Text.assemble(
                    (str(self.file_removals), "red"),
                    (
                        f" {simple_pluralise('file', self.file_removals)} removed",
                        "red",
                    ),
                )
            )

        bar_width = console.width // 5
        changed_lines = max(1, self.line_additions + self.line_removals)
        added_lines_ratio = self.line_additions / changed_lines

        line_changes_summary = Table.grid()
        line_changes_summary.add_column()
        line_changes_summary.add_column()
        line_changes_summary.add_column()
        line_changes_summary.add_row(
            Text(f"+{self.line_additions}", style="bold green"),
            UnderlineBar(
                highlight_range=(0, added_lines_ratio * bar_width),
                highlight_style="green",
                background_style="red",
                width=bar_width,
            ),
            Text(f"-{self.line_removals}", style="bold red"),
        )

        bar_hpad = len(str(self.line_additions)) + len(str(self.line_removals)) + 4
        yield Align.center(line_changes_summary, width=bar_width + bar_hpad)
        yield Text("\n")


class RemovedFileBody:
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        yield Rule(Text(" File was removed ", style="red"), characters="╲", style="hatched")
        yield Rule(characters="╲", style="hatched")
        yield Rule(style="border", characters="▔")


@dataclass
class RestrictedSourceNotice:
    """Shown when a target file cannot be safely bound to the work tree."""

    reason: str

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(
            Text(
                f" restricted source: {self.reason} — showing only lines contained in the patch ",
                style="yellow",
            ),
            characters="╲",
            style="hatched",
        )


@dataclass
class BinaryFileBody:
    size_in_bytes: Optional[int]
    restricted: bool = False

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        if self.restricted:
            label = Text(
                " File is binary · size unavailable (restricted source) ",
                style="yellow",
            )
        else:
            label = Text(
                f" File is binary · {self.size_in_bytes} bytes ", style="blue"
            )
        yield Rule(label, characters="╲", style="hatched")
        yield Rule(characters="╲", style="hatched")
        yield Rule(style="border", characters="▔")


class PatchedFileHeader:
    def __init__(self, patch: PatchedFile):
        self.patch = patch

        target_name = self._display_name(patch.target_file)
        self.target_name = target_name

        if patch.is_rename:
            self.source_name = Path(self._display_name(patch.source_file)).name
        else:
            self.source_name = None

    @staticmethod
    def _display_name(raw_file: str) -> str:
        decoded = decode_git_path(raw_file)
        if decoded is None:
            # Malformed quoting -- display the raw value, controls are made
            # visible by the renderer.
            return raw_file
        return strip_git_prefix(decoded)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        title = Text()
        if self.source_name is not None:
            title.append_text(sanitize_text(self.source_name, style="dim strike"))
            title.append(" → ", style="dim")
        elif self.patch.is_added_file:
            title.append("Added ", style="bold green")

        title.append_text(sanitize_text(self.target_name))
        title.append(" (")
        title.append(f"{self.patch.added} additions", style="green")
        title.append(", ")
        title.append(f"{self.patch.removed} removals", style="red")
        title.append(")")

        yield Rule(title, style="border", characters="▁")


class OnlyRenamedFileBody:
    """Represents a file that was renamed but the content was not changed."""

    def __init__(self, patch: PatchedFile):
        self.patch = patch

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        yield Rule(
            Text(" File was only renamed ", style="blue"),
            characters="╲",
            style="hatched",
        )
        yield Rule(characters="╲", style="hatched")
        yield Rule(style="border", characters="▔")
