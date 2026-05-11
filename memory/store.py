import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from memory.schema import PRAnalysisRecord, RepoContext
from config.settings import MEMORY_DB_PATH, MEMORY_MAX_HISTORY_ENTRIES


class MemoryStore:
    """
    Lightweight SQLite store for PR analysis history.

    Two tables:
      pr_analyses   — one row per analyzed PR
      repo_stats    — one row per repo (upserted), aggregated stats

    Why SQLite instead of ChromaDB?
    We're storing structured JSON records, not vectors. SQLite is zero-config,
    runs in-process, and is much faster to set up. Swap to Postgres or
    ChromaDB later if you need vector similarity search over PR history.
    """

    def __init__(self, db_path: str = MEMORY_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row   # access columns by name
        self._create_tables()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS pr_analyses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                repo            TEXT NOT NULL,
                pr_number       INTEGER NOT NULL,
                pr_title        TEXT,
                analyzed_at     TEXT NOT NULL,
                files_touched   TEXT,   -- JSON list
                symbols_changed TEXT,   -- JSON list
                languages       TEXT,   -- JSON list
                additions       INTEGER DEFAULT 0,
                deletions       INTEGER DEFAULT 0,
                overall_risk_score  REAL DEFAULT 0.0,
                risk_level          TEXT DEFAULT 'UNKNOWN',
                blast_radius_count  INTEGER DEFAULT 0,
                had_test_gaps       INTEGER DEFAULT 0,
                top_concerns        TEXT,   -- JSON list
                critic_verdict      TEXT DEFAULT 'AGREE',
                conflict_rounds     INTEGER DEFAULT 0,
                confirmed_incident  INTEGER,
                incident_notes      TEXT,
                UNIQUE(repo, pr_number)
            );

            CREATE TABLE IF NOT EXISTS repo_stats (
                repo                    TEXT PRIMARY KEY,
                total_prs_analyzed      INTEGER DEFAULT 0,
                avg_risk_score          REAL DEFAULT 0.0,
                high_risk_prs           INTEGER DEFAULT 0,
                most_touched_files      TEXT,   -- JSON list
                recurring_test_gap_files TEXT,  -- JSON list
                last_updated            TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_pr_repo ON pr_analyses(repo);
            CREATE INDEX IF NOT EXISTS idx_pr_risk  ON pr_analyses(overall_risk_score);
        """)
        self.conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_analysis(self, record: PRAnalysisRecord):
        """
        Insert or replace a PR analysis record.
        Called by the report synthesizer after a run completes.
        """
        self.conn.execute("""
            INSERT INTO pr_analyses (
                repo, pr_number, pr_title, analyzed_at,
                files_touched, symbols_changed, languages,
                additions, deletions,
                overall_risk_score, risk_level, blast_radius_count,
                had_test_gaps, top_concerns, critic_verdict, conflict_rounds
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(repo, pr_number) DO UPDATE SET
                overall_risk_score  = excluded.overall_risk_score,
                risk_level          = excluded.risk_level,
                blast_radius_count  = excluded.blast_radius_count,
                had_test_gaps       = excluded.had_test_gaps,
                top_concerns        = excluded.top_concerns,
                critic_verdict      = excluded.critic_verdict,
                conflict_rounds     = excluded.conflict_rounds,
                analyzed_at         = excluded.analyzed_at
        """, (
            record.repo,
            record.pr_number,
            record.pr_title,
            record.analyzed_at,
            json.dumps(record.files_touched),
            json.dumps(record.symbols_changed),
            json.dumps(record.languages),
            record.additions,
            record.deletions,
            record.overall_risk_score,
            record.risk_level,
            record.blast_radius_count,
            int(record.had_test_gaps),
            json.dumps(record.top_concerns),
            record.critic_verdict,
            record.conflict_rounds,
        ))
        self.conn.commit()
        self._update_repo_stats(record.repo)

    def _update_repo_stats(self, repo: str):
        """Recompute and upsert aggregate stats for a repo."""
        rows = self.conn.execute(
            "SELECT * FROM pr_analyses WHERE repo = ?", (repo,)
        ).fetchall()

        if not rows:
            return

        total = len(rows)
        avg_risk = sum(r["overall_risk_score"] for r in rows) / total
        high_risk = sum(1 for r in rows if r["overall_risk_score"] >= 7.0)

        # Most touched files
        file_counts: dict = {}
        for row in rows:
            for f in json.loads(row["files_touched"] or "[]"):
                file_counts[f] = file_counts.get(f, 0) + 1
        most_touched = sorted(file_counts, key=file_counts.get, reverse=True)[:5]

        # Files that repeatedly had test gaps
        gap_files: dict = {}
        for row in rows:
            if row["had_test_gaps"]:
                for f in json.loads(row["files_touched"] or "[]"):
                    gap_files[f] = gap_files.get(f, 0) + 1
        recurring_gaps = sorted(gap_files, key=gap_files.get, reverse=True)[:5]

        self.conn.execute("""
            INSERT INTO repo_stats (
                repo, total_prs_analyzed, avg_risk_score, high_risk_prs,
                most_touched_files, recurring_test_gap_files, last_updated
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(repo) DO UPDATE SET
                total_prs_analyzed      = excluded.total_prs_analyzed,
                avg_risk_score          = excluded.avg_risk_score,
                high_risk_prs           = excluded.high_risk_prs,
                most_touched_files      = excluded.most_touched_files,
                recurring_test_gap_files= excluded.recurring_test_gap_files,
                last_updated            = excluded.last_updated
        """, (
            repo, total, round(avg_risk, 2), high_risk,
            json.dumps(most_touched), json.dumps(recurring_gaps),
            datetime.now(timezone.utc).isoformat(),
        ))
        self.conn.commit()

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_repo_context(self, repo: str) -> RepoContext | None:
        """
        Returns aggregated context for a repo — injected into agent prompts
        so agents know the repo's historical risk patterns.
        Returns None if repo has never been analyzed.
        """
        stats_row = self.conn.execute(
            "SELECT * FROM repo_stats WHERE repo = ?", (repo,)
        ).fetchone()

        if not stats_row:
            return None

        # Fetch recent PR summaries (last N)
        recent_rows = self.conn.execute(
            """SELECT pr_number, pr_title, overall_risk_score, risk_level, analyzed_at
               FROM pr_analyses WHERE repo = ?
               ORDER BY analyzed_at DESC LIMIT ?""",
            (repo, MEMORY_MAX_HISTORY_ENTRIES)
        ).fetchall()

        summaries = [
            f"PR #{r['pr_number']} '{r['pr_title']}' → {r['risk_level']} ({r['overall_risk_score']:.1f}/10)"
            for r in recent_rows
        ]

        return RepoContext(
            repo=repo,
            total_prs_analyzed=stats_row["total_prs_analyzed"],
            avg_risk_score=stats_row["avg_risk_score"],
            high_risk_prs=stats_row["high_risk_prs"],
            most_touched_files=json.loads(stats_row["most_touched_files"] or "[]"),
            recurring_test_gap_files=json.loads(stats_row["recurring_test_gap_files"] or "[]"),
            recent_summaries=summaries,
        )

    def get_pr_record(self, repo: str, pr_number: int) -> PRAnalysisRecord | None:
        """Retrieve a specific PR's analysis record."""
        row = self.conn.execute(
            "SELECT * FROM pr_analyses WHERE repo = ? AND pr_number = ?",
            (repo, pr_number)
        ).fetchone()

        if not row:
            return None

        return PRAnalysisRecord(
            repo=row["repo"],
            pr_number=row["pr_number"],
            pr_title=row["pr_title"] or "",
            analyzed_at=row["analyzed_at"],
            files_touched=json.loads(row["files_touched"] or "[]"),
            symbols_changed=json.loads(row["symbols_changed"] or "[]"),
            languages=json.loads(row["languages"] or "[]"),
            additions=row["additions"],
            deletions=row["deletions"],
            overall_risk_score=row["overall_risk_score"],
            risk_level=row["risk_level"],
            blast_radius_count=row["blast_radius_count"],
            had_test_gaps=bool(row["had_test_gaps"]),
            top_concerns=json.loads(row["top_concerns"] or "[]"),
            critic_verdict=row["critic_verdict"],
            conflict_rounds=row["conflict_rounds"],
        )

    def format_context_for_prompt(self, repo: str) -> str:
        """
        Returns a human-readable summary of repo history
        ready to paste into an agent prompt.
        """
        ctx = self.get_repo_context(repo)
        if not ctx:
            return "No prior analysis history for this repository."

        lines = [
            f"Repository: {ctx.repo}",
            f"PRs analyzed: {ctx.total_prs_analyzed}  |  Avg risk: {ctx.avg_risk_score:.1f}/10  |  High-risk PRs: {ctx.high_risk_prs}",
            f"Most frequently changed files: {', '.join(ctx.most_touched_files) or 'none'}",
            f"Files with recurring test gaps: {', '.join(ctx.recurring_test_gap_files) or 'none'}",
            "",
            "Recent PR history:",
        ] + [f"  • {s}" for s in ctx.recent_summaries[:5]]

        return "\n".join(lines)


    def get_historical_context(self, repo: str, changed_files: list, changed_symbols: list) -> dict:
        """
        Enhancement 5: Structured historical context retrieval.

        Returns a dict ready to inject into agent prompts and the final report:
        {
          "past_high_risk_prs": int,
          "avg_risk_score": float,
          "frequently_affected_modules": [str],
          "files_with_recurring_issues": [str],
          "repeated_failure_symbols": [str],
          "historical_risk_trend": "IMPROVING|STABLE|WORSENING",
          "prompt_text": str   ← formatted for direct injection into prompts
        }

        Uses SQLite only — no vector search needed.
        """
        rows = self.conn.execute(
            "SELECT * FROM pr_analyses WHERE repo = ? ORDER BY analyzed_at DESC LIMIT 50",
            (repo,)
        ).fetchall()

        if not rows:
            return {
                "past_high_risk_prs": 0,
                "avg_risk_score": 0.0,
                "frequently_affected_modules": [],
                "files_with_recurring_issues": [],
                "repeated_failure_symbols": [],
                "historical_risk_trend": "STABLE",
                "prompt_text": "No prior analysis history for this repository.",
            }

        import json as _json

        # ── Basic stats ────────────────────────────────────────────────────────
        scores = [r["overall_risk_score"] for r in rows]
        avg    = sum(scores) / len(scores)
        high   = sum(1 for s in scores if s >= 7.0)

        # ── File recurrence: which files appear in many past PRs ───────────────
        file_counts: dict = {}
        for row in rows:
            for f in _json.loads(row["files_touched"] or "[]"):
                file_counts[f] = file_counts.get(f, 0) + 1
        recurring_files = [f for f, c in sorted(file_counts.items(), key=lambda x: -x[1]) if c >= 2][:5]

        # ── Symbol recurrence across past PRs ─────────────────────────────────
        symbol_counts: dict = {}
        for row in rows:
            for s in _json.loads(row["symbols_changed"] or "[]"):
                symbol_counts[s] = symbol_counts.get(s, 0) + 1
        recurring_symbols = [s for s, c in sorted(symbol_counts.items(), key=lambda x: -x[1]) if c >= 2][:5]

        # ── Module frequency (directory-level grouping) ────────────────────────
        module_counts: dict = {}
        for f in file_counts:
            parts = f.split("/")
            if len(parts) >= 2:
                module = parts[0] if parts[0] not in ("src", "lib", "app") else parts[1]
            else:
                module = parts[0]
            module_counts[module] = module_counts.get(module, 0) + file_counts[f]
        frequent_modules = [m for m, _ in sorted(module_counts.items(), key=lambda x: -x[1])][:5]

        # ── Files with recurring gaps (had_test_gaps in 2+ PRs) ───────────────
        gap_file_counts: dict = {}
        for row in rows:
            if row["had_test_gaps"]:
                for f in _json.loads(row["files_touched"] or "[]"):
                    gap_file_counts[f] = gap_file_counts.get(f, 0) + 1
        gap_files = [f for f, c in sorted(gap_file_counts.items(), key=lambda x: -x[1]) if c >= 2][:3]

        # ── Risk trend: compare last 5 vs previous 5 ──────────────────────────
        trend = "STABLE"
        if len(scores) >= 10:
            recent_avg = sum(scores[:5]) / 5
            older_avg  = sum(scores[5:10]) / 5
            if recent_avg > older_avg + 1.0:
                trend = "WORSENING"
            elif recent_avg < older_avg - 1.0:
                trend = "IMPROVING"

        # ── Files overlap: how many current files appeared in past high-risk PRs ──
        high_risk_files: set = set()
        for row in rows:
            if row["overall_risk_score"] >= 7.0:
                for f in _json.loads(row["files_touched"] or "[]"):
                    high_risk_files.add(f)
        overlapping = [f for f in changed_files if f in high_risk_files]

        # ── Symbol overlap with past PRs ──────────────────────────────────────
        overlapping_symbols = [s for s in changed_symbols if s in symbol_counts and symbol_counts[s] >= 2]

        # ── Build prompt text ──────────────────────────────────────────────────
        lines = [
            f"HISTORICAL CONTEXT FOR {repo}:",
            f"  PRs analyzed: {len(rows)}  |  Avg risk: {avg:.1f}/10  |  High-risk PRs: {high}  |  Trend: {trend}",
        ]
        if frequent_modules:
            lines.append(f"  Most-changed modules: {', '.join(frequent_modules)}")
        if recurring_files:
            lines.append(f"  Files changed repeatedly: {', '.join(recurring_files[:3])}")
        if gap_files:
            lines.append(f"  Files with recurring test gaps: {', '.join(gap_files)}")
        if overlapping:
            lines.append(f"  ⚠ CURRENT PR touches files that appeared in past high-risk PRs: {', '.join(overlapping[:3])}")
        if overlapping_symbols:
            lines.append(f"  ⚠ CURRENT PR changes symbols seen in multiple past PRs: {', '.join(overlapping_symbols[:3])}")

        return {
            "past_high_risk_prs":           high,
            "avg_risk_score":               round(avg, 2),
            "frequently_affected_modules":  frequent_modules,
            "files_with_recurring_issues":  recurring_files,
            "repeated_failure_symbols":     recurring_symbols,
            "historical_risk_trend":        trend,
            "files_overlapping_past_high_risk": overlapping,
            "symbols_seen_before":          overlapping_symbols,
            "prompt_text":                  "\n".join(lines),
        }

    def close(self):
        self.conn.close()
