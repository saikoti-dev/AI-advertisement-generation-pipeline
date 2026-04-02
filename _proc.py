"""
_proc.py
--------
Spawns a subprocess for a model task, streams its output live,
and returns the parsed RESULT_JSON result dict.

Key behaviours:
- stdout and stderr are kept on SEPARATE pipes (no double-printing)
- stderr is drained in a background thread (prevents deadlock)
- RESULT_JSON sentinel is consumed silently — not printed
- Repetitive step/progress lines are COLLAPSED: the log panel sees
  one updating line instead of 50 "[VideoGen] step N/50" entries
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

_RUNNER = str(Path(__file__).parent / "_subprocess_runner.py")

# Line prefixes that are "overwrite" lines — only the latest value matters.
# Instead of appending 50 new log entries, _proc replaces the previous one
# in-place by printing a carriage-return before the line. The parent process's
# _TeeStream in pipeline_logger.py treats \r as an in-place overwrite.
_OVERWRITE_PREFIXES = (
    "[VideoGen] step ",
    "[progress]",
    "[mem:",
    "[device_map]",
)


def _is_overwrite(line: str) -> bool:
    return any(line.startswith(p) for p in _OVERWRITE_PREFIXES)


def run_task(task: str, args: dict, timeout: int = 900) -> dict:
    """
    Spawn a subprocess for `task`, stream output live, return result dict.
    Raises RuntimeError on non-zero exit or missing RESULT_JSON sentinel.
    """
    payload = json.dumps(args)

    print(f"\n{'─'*56}", flush=True)
    print(f"[proc] task={task!r}", flush=True)
    print(f"{'─'*56}", flush=True)

    proc = subprocess.Popen(
        [sys.executable, _RUNNER, task, payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,   # separate — prevents tqdm double-print
        text=True,
        bufsize=1,
    )

    result_line: str | None = None
    output_lines: list[str] = []
    stderr_lines: list[str] = []

    # ── Drain stderr silently into a buffer (prevents pipe deadlock) ──────────
    # We don't print stderr to avoid double-printing tqdm bars.
    # On task failure it's included in the error message.
    def _drain_stderr():
        for line in proc.stderr:
            s = line.rstrip("\n")
            if s:
                stderr_lines.append(s)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # ── Stream stdout live ────────────────────────────────────────────────────
    last_overwrite_prefix: str | None = None   # track which prefix we're collapsing

    for line in proc.stdout:
        s = line.rstrip("\n")
        output_lines.append(s)

        if not s:
            continue

        if s.startswith("RESULT_JSON:"):
            result_line = s   # internal sentinel — never printed
            continue

        if _is_overwrite(s):
            # Determine the "group" for this overwrite line so that a
            # "[VideoGen] step" line doesn't clobber a "[mem:" line
            current_prefix = next(p for p in _OVERWRITE_PREFIXES if s.startswith(p))

            if current_prefix == last_overwrite_prefix:
                # Same group — overwrite the previous line in the terminal
                # (and in the Streamlit log via _TeeStream's \r handling)
                print(f"\r{s}", end="", flush=True)
            else:
                # New group — start a fresh line first
                if last_overwrite_prefix is not None:
                    print()   # close the previous overwrite line
                print(f"\r{s}", end="", flush=True)
                last_overwrite_prefix = current_prefix
        else:
            # Normal line — if we were in overwrite mode, close it first
            if last_overwrite_prefix is not None:
                print()   # newline after the last overwrite line
                last_overwrite_prefix = None
            print(s, flush=True)

    # Close any dangling overwrite line
    if last_overwrite_prefix is not None:
        print(flush=True)

    proc.wait()
    stderr_thread.join(timeout=5)

    print(f"{'─'*56}", flush=True)
    print(f"[proc] exited  code={proc.returncode}  task={task!r}", flush=True)
    print(f"{'─'*56}\n", flush=True)

    if proc.returncode != 0:
        sig_hint = ""
        if proc.returncode == -9:
            sig_hint = (
                "\n⚠ Exit code -9 = SIGKILL (Linux OOM killer — system RAM full).\n"
                "  Restart the Colab runtime to reclaim RAM, then rerun."
            )
        combined = output_lines + stderr_lines
        raise RuntimeError(
            f"Subprocess task={task!r} failed (exit {proc.returncode}).{sig_hint}\n"
            f"Last 20 lines:\n" + "\n".join(combined[-20:])
        )

    if result_line is None:
        raise RuntimeError(
            f"Subprocess task={task!r} exited 0 but never printed RESULT_JSON.\n"
            f"Full stdout:\n" + "\n".join(output_lines)
        )

    return json.loads(result_line[len("RESULT_JSON:"):])
