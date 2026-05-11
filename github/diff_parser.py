import re
from dataclasses import dataclass, field
from pathlib import Path
from config.settings import SUPPORTED_LANGUAGES, SYMBOL_PATTERNS


@dataclass
class ChangedSymbol:
    """A function, class, or interface that was modified."""
    name: str
    kind: str           # "function" | "class" | "interface" | "method"
    change_type: str    # "added" | "removed" | "modified"
    line_number: int    # approximate line in the diff hunk


@dataclass
class ChangedFile:
    """Represents a single file's changes within a PR diff."""
    path: str
    language: str
    change_type: str            # "modified" | "added" | "deleted" | "renamed"
    old_path: str               # populated for renames
    additions: int
    deletions: int
    changed_symbols: list[ChangedSymbol] = field(default_factory=list)
    diff_hunks: list[str] = field(default_factory=list)   # raw hunk text
    raw_diff: str = ""

    @property
    def is_test_file(self) -> bool:
        p = self.path.lower()
        return any(seg in p for seg in [
            "test", "spec", "__tests__", "_test.", ".test.", ".spec."
        ])

    @property
    def is_config_file(self) -> bool:
        return Path(self.path).suffix in {
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env"
        }


@dataclass
class ParsedDiff:
    """Top-level result of parsing a full PR diff."""
    changed_files: list[ChangedFile] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    languages: list[str] = field(default_factory=list)

    # Flattened lists — useful for agent prompts
    all_changed_symbols: list[str] = field(default_factory=list)
    test_files_changed: list[str] = field(default_factory=list)
    source_files_changed: list[str] = field(default_factory=list)

    @property
    def has_test_changes(self) -> bool:
        return len(self.test_files_changed) > 0

    @property
    def summary(self) -> str:
        return (
            f"{len(self.changed_files)} files changed "
            f"(+{self.total_additions} / -{self.total_deletions}), "
            f"languages: {', '.join(self.languages) or 'unknown'}, "
            f"symbols changed: {', '.join(self.all_changed_symbols[:10]) or 'none detected'}"
        )

    def to_agent_context(self) -> dict:
        """Serialize to a flat dict suitable for injecting into agent prompts."""
        return {
            "diff_summary": self.summary,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "changed_files": [f.path for f in self.changed_files],
            "source_files": self.source_files_changed,
            "test_files": self.test_files_changed,
            "changed_symbols": self.all_changed_symbols,
            "languages": self.languages,
            "has_test_changes": self.has_test_changes,
            "per_file": [
                {
                    "path": f.path,
                    "language": f.language,
                    "change_type": f.change_type,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "symbols": [s.name for s in f.changed_symbols],
                    "is_test": f.is_test_file,
                }
                for f in self.changed_files
            ],
        }


# ── Parser ─────────────────────────────────────────────────────────────────────

class DiffParser:
    """
    Parses a unified diff (git diff / GitHub PR diff format) into a ParsedDiff.

    Unified diff format recap:
      diff --git a/path b/path
      index abc..def 100644
      --- a/path
      +++ b/path
      @@ -start,count +start,count @@ optional_context
      -removed line
      +added line
       context line
    """

    # Matches the file header line
    _FILE_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
    # Matches rename
    _RENAME_FROM = re.compile(r"^rename from (.+)$")
    _RENAME_TO = re.compile(r"^rename to (.+)$")
    # Matches hunk header
    _HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    # New file / deleted file markers
    _NEW_FILE = re.compile(r"^new file mode")
    _DEL_FILE = re.compile(r"^deleted file mode")

    def parse(self, raw_diff: str) -> ParsedDiff:
        """Entry point — parse a full diff string into a ParsedDiff."""
        result = ParsedDiff()
        current_file: ChangedFile | None = None
        current_hunk_lines: list[str] = []
        current_hunk_start_line = 0
        rename_from = None

        for line in raw_diff.splitlines():

            # ── New file block ─────────────────────────────────────────────
            m = self._FILE_HEADER.match(line)
            if m:
                # Save the previous file
                if current_file is not None:
                    if current_hunk_lines:
                        current_file.diff_hunks.append("\n".join(current_hunk_lines))
                    self._finalise_file(current_file, result)

                path_b = m.group(2)
                current_file = ChangedFile(
                    path=path_b,
                    language=self._detect_language(path_b),
                    change_type="modified",
                    old_path=path_b,
                    additions=0,
                    deletions=0,
                )
                current_hunk_lines = []
                rename_from = None
                continue

            if current_file is None:
                continue

            # ── File-level markers ─────────────────────────────────────────
            if self._NEW_FILE.match(line):
                current_file.change_type = "added"
                continue

            if self._DEL_FILE.match(line):
                current_file.change_type = "deleted"
                continue

            m = self._RENAME_FROM.match(line)
            if m:
                rename_from = m.group(1)
                continue

            m = self._RENAME_TO.match(line)
            if m:
                current_file.change_type = "renamed"
                current_file.old_path = rename_from or current_file.path
                current_file.path = m.group(1)
                continue

            # ── Hunk header ────────────────────────────────────────────────
            m = self._HUNK_HEADER.match(line)
            if m:
                if current_hunk_lines:
                    current_file.diff_hunks.append("\n".join(current_hunk_lines))
                current_hunk_lines = [line]
                current_hunk_start_line = int(m.group(1))
                continue

            # ── Diff content lines ─────────────────────────────────────────
            if line.startswith("+") and not line.startswith("+++"):
                current_file.additions += 1
                current_hunk_lines.append(line)

                # Check if this added line defines a symbol
                symbol = self._extract_symbol(line[1:], current_file.language)
                if symbol:
                    current_file.changed_symbols.append(ChangedSymbol(
                        name=symbol,
                        kind=self._classify_symbol(line[1:], current_file.language),
                        change_type="added",
                        line_number=current_hunk_start_line + current_file.additions,
                    ))

            elif line.startswith("-") and not line.startswith("---"):
                current_file.deletions += 1
                current_hunk_lines.append(line)

                symbol = self._extract_symbol(line[1:], current_file.language)
                if symbol:
                    current_file.changed_symbols.append(ChangedSymbol(
                        name=symbol,
                        kind=self._classify_symbol(line[1:], current_file.language),
                        change_type="removed",
                        line_number=current_hunk_start_line,
                    ))
            else:
                current_hunk_lines.append(line)

        # Save the last file
        if current_file is not None:
            if current_hunk_lines:
                current_file.diff_hunks.append("\n".join(current_hunk_lines))
            self._finalise_file(current_file, result)

        # Build top-level aggregates
        result.total_additions = sum(f.additions for f in result.changed_files)
        result.total_deletions = sum(f.deletions for f in result.changed_files)
        result.languages = list({f.language for f in result.changed_files if f.language != "unknown"})

        all_symbols: list[str] = []
        for f in result.changed_files:
            for s in f.changed_symbols:
                if s.name not in all_symbols:
                    all_symbols.append(s.name)
        result.all_changed_symbols = all_symbols

        result.test_files_changed = [f.path for f in result.changed_files if f.is_test_file]
        result.source_files_changed = [f.path for f in result.changed_files if not f.is_test_file]

        # Attach raw diff per file (useful for agent prompts)
        self._attach_raw_diff(raw_diff, result)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _finalise_file(self, f: ChangedFile, result: ParsedDiff):
        result.changed_files.append(f)

    def _detect_language(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return SUPPORTED_LANGUAGES.get(ext, "unknown")

    def _extract_symbol(self, line: str, language: str) -> str | None:
        """Return the first symbol name found in a line, or None."""
        patterns = SYMBOL_PATTERNS.get(language, [])
        for pattern in patterns:
            m = re.search(pattern, line.strip())
            if m:
                return m.group(1)
        return None

    def _classify_symbol(self, line: str, language: str) -> str:
        """Rough classification of a symbol as function/class/interface."""
        line = line.strip()
        if re.search(r"\bclass\b", line):
            return "class"
        if re.search(r"\binterface\b", line):
            return "interface"
        if re.search(r"\bstruct\b", line):
            return "struct"
        return "function"

    def _attach_raw_diff(self, raw_diff: str, result: ParsedDiff):
        """
        Attach the raw diff slice for each file — used by the Change Simulator
        to show the agent the exact before/after without re-parsing.
        """
        blocks: dict[str, list[str]] = {}
        current_path = None

        for line in raw_diff.splitlines():
            m = self._FILE_HEADER.match(line)
            if m:
                current_path = m.group(2)
                blocks[current_path] = []
            if current_path:
                blocks[current_path].append(line)

        for f in result.changed_files:
            if f.path in blocks:
                f.raw_diff = "\n".join(blocks[f.path])


# ── Convenience function ───────────────────────────────────────────────────────

def parse_diff(raw_diff: str) -> ParsedDiff:
    """Shorthand — parse a raw diff string and return a ParsedDiff."""
    return DiffParser().parse(raw_diff)