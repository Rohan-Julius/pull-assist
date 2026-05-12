"""
Pixel art banner displayed when the CLI starts up.

Displays a pull request merge icon (matching the pink→yellow gradient logo)
followed by the PULL ASSIST wordmark. Inspired by GitHub Copilot's startup.
"""

import os
import shutil
from rich.console import Console
from rich.text import Text


# ── Gradient color palette (pink → coral → orange → yellow) ───────────────────
# Each row of the ASCII art gets a color from this gradient.
GRADIENT = [
    "#FF69B4",  # Hot pink
    "#FF6BA8",
    "#FF6D9C",
    "#FF7090",
    "#FF7584",
    "#FF7A78",
    "#FF806C",
    "#FF8860",
    "#FF9054",
    "#FF9848",
    "#FFA03C",
    "#FFAA30",
    "#FFB424",
    "#FFBE18",
    "#FFC80C",
    "#FFD700",  # Gold/Yellow
]

# ── PR Merge Icon — pixel art ─────────────────────────────────────────────────
# Represents the pull request merge symbol from the attached logo image:
#   Two branches with circles, an arrow merging right→left
#   Pink at top, fading to yellow at bottom
# ALL LINES MUST BE THE SAME LENGTH to prevent centering distortion.
_ICON_W = 30
PR_ICON_LINES = [
    "      ▄▀▀▀▄                ",
    "      █   █  ◀━━━━━━┓      ",
    "      ▀▄▄▄▀         ┃      ",
    "        ┃           ┃      ",
    "        ┃           ┃      ",
    "        ┃           ┃      ",
    "        ┃           ┃      ",
    "      ▄▀▀▀▄       ▄▀▀▀▄   ",
    "      █   █       █   █   ",
    "      ▀▄▄▄▀       ▀▄▄▄▀   ",
]

# ── PULL ASSIST wordmark (compact, fits 80-col terminals) ────────────────────
# ALL LINES MUST BE THE SAME LENGTH.
_WORD_W = 56
WORDMARK_LINES = [
    " ████  █  █ █    █         █████ ████ ████ █ ████ █████",
    " █  █  █  █ █    █        █▀  █▀ █    █    █ █      █ ",
    " ████  █  █ █    █        █████  ███  ███  █ ███    █ ",
    " █     █  █ █    █        █▀  █▀    █    █ █    █   █ ",
    " █     ████ ████ ████     █   █  ████ ████ █ ████   █ ",
]

TAGLINE = "AI-powered PR impact analysis · Risk · Blast radius · Deploy"
VERSION_LINE = "v0.1.0 -- github.com/Rohan-Julius/pull-assist"


def _gradient_color(index: int, total: int) -> str:
    """Pick a gradient color based on position (0=pink, total=yellow)."""
    if total <= 1:
        return GRADIENT[0]
    pos = int((index / (total - 1)) * (len(GRADIENT) - 1))
    pos = min(pos, len(GRADIENT) - 1)
    return GRADIENT[pos]


def _center_pad(line: str, term_width: int) -> str:
    """Manually center a line by adding leading spaces. Avoids Rich's
    justify='center' which miscounts Unicode character widths."""
    # Use raw character count (not display width) for consistent alignment
    pad = max(0, (term_width - len(line)) // 2)
    return " " * pad + line


def _print_gradient_block(console: Console, lines: list[str],
                          offset: int = 0, total: int = None,
                          term_width: int = 80):
    """Print a block of lines with a pink→yellow vertical gradient, manually centered."""
    if total is None:
        total = len(lines)
    for i, line in enumerate(lines):
        color = _gradient_color(i + offset, total)
        centered = _center_pad(line, term_width)
        text = Text(centered)
        text.stylize(f"bold {color}")
        console.print(text)


def print_banner(console: Console = None, compact: bool = False):
    """
    Print the startup banner with the pink→yellow gradient PR icon + wordmark.

    Args:
        console: Rich Console instance. Creates one if not provided.
        compact: If True, skip the PR icon and only show the wordmark.
    """
    if console is None:
        console = Console()

    # Get terminal width for centering
    tw = shutil.get_terminal_size((80, 24)).columns

    console.print()

    total_lines = len(PR_ICON_LINES) + 1 + len(WORDMARK_LINES)

    if not compact:
        _print_gradient_block(console, PR_ICON_LINES, offset=0,
                              total=total_lines, term_width=tw)
        console.print()

    word_offset = len(PR_ICON_LINES) + 1 if not compact else 0
    word_total = total_lines if not compact else len(WORDMARK_LINES)
    _print_gradient_block(console, WORDMARK_LINES, offset=word_offset,
                          total=word_total, term_width=tw)

    console.print()
    # Center tagline and version with plain spaces (no Rich justify)
    console.print(Text(_center_pad(TAGLINE, tw), style="dim"))
    console.print(Text(_center_pad(VERSION_LINE, tw), style="dim"))
    console.print()


def print_mini_banner(console: Console = None):
    """Single-line banner for subcommand headers."""
    if console is None:
        console = Console()

    parts = [
        ("◉ ", "#FF69B4"),
        ("pull", "#FF8860"),
        ("-", "#FFB424"),
        ("assist", "#FFD700"),
    ]

    text = Text()
    for part_text, color in parts:
        text.append(part_text, style=f"bold {color}")

    text.append(" v0.1.0", style="dim")
    console.print(text)
    console.print()
