"""
pipeline_logger.py
------------------
Captures ALL pipeline output into a caller-supplied list so Streamlit
can read it live on every rerun — no copy-at-end needed.

Key design: the log list is passed IN by the caller (e.g. st.session_state.log_lines).
The background thread appends directly to that same object, so every Streamlit
rerun sees the latest lines immediately.

HuggingFace download bars use \r (carriage return) instead of \n to overwrite
the same terminal line. We treat \r as a line boundary and OVERWRITE the last
[progress] line in the list — so the log shows a single updating download bar
rather than flooding with hundreds of duplicate lines.
"""

from __future__ import annotations

import io
import re
import sys
import threading
from typing import List

_ANSI = re.compile(r'\x1b\[[0-9;?]*[mGKHFJA-Za-z]')


def _strip_ansi(s: str) -> str:
    return _ANSI.sub('', s)


class _TeeStream(io.TextIOBase):
    """
    Writes to the original stream AND appends complete lines to log_list.
    Handles both \n-terminated lines (normal print) and \r-terminated lines
    (HuggingFace download bars / tqdm carriage-return updates).
    """

    def __init__(self, original, log_list: List[str], lock: threading.Lock):
        self._original = original
        self._log = log_list
        self._lock = lock
        self._buf = ""

    def _emit(self, raw: str, overwrite: bool):
        """Add a cleaned line to the log. If overwrite=True, replace the last [progress] line."""
        line = _strip_ansi(raw).strip()
        if not line:
            return
        with self._lock:
            if overwrite and self._log and self._log[-1].startswith("[progress]"):
                # Update in-place: the download bar moved forward
                self._log[-1] = f"[progress] {line}"
            else:
                prefix = "[progress] " if overwrite else ""
                self._log.append(f"{prefix}{line}")

    def write(self, s: str) -> int:
        if not s:
            return 0
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass

        self._buf += s

        # Process all complete segments (split on \n or \r)
        while True:
            ni = self._buf.find('\n')
            ri = self._buf.find('\r')

            if ni == -1 and ri == -1:
                break  # no complete line yet

            if ri != -1 and (ni == -1 or ri < ni):
                # \r comes first — carriage-return overwrite (HF download / tqdm)
                segment = self._buf[:ri]
                self._buf = self._buf[ri + 1:]
                self._emit(segment, overwrite=True)
            else:
                # \n comes first — normal newline
                segment = self._buf[:ni]
                self._buf = self._buf[ni + 1:]
                self._emit(segment.rstrip('\r'), overwrite=False)

        return len(s)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def fileno(self):
        return self._original.fileno()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "replace")

    def isatty(self):
        return False


class PipelineLogger:
    """
    Pass a mutable list (e.g. st.session_state.log_lines) as `log_list`.
    The background thread appends directly to it — Streamlit reads the
    same object on every loop iteration, giving live updates with zero extra plumbing.

    Usage:
        logger = PipelineLogger(st.session_state.log_lines)
        ...pipeline runs, appending to log_lines live...
        logger.stop()
    """

    def __init__(self, log_list: List[str]):
        self._lines = log_list          # shared reference — NOT a copy
        self._lock = threading.Lock()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._saved = {}                # original tqdm classes keyed by module path

        sys.stdout = _TeeStream(self._orig_stdout, self._lines, self._lock)
        sys.stderr = _TeeStream(self._orig_stderr, self._lines, self._lock)
        self._patch_tqdm()

    # ── public API ──────────────────────────────────────────────────────────

    def add(self, msg: str):
        """Manually inject a line (e.g. step headers)."""
        with self._lock:
            self._lines.append(msg)

    def stop(self):
        """Flush partial buffer, restore stdout/stderr and tqdm."""
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, _TeeStream) and stream._buf:
                line = _strip_ansi(stream._buf).strip()
                if line:
                    with self._lock:
                        self._lines.append(line)
                stream._buf = ""

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._unpatch_tqdm()

    # ── tqdm patching ────────────────────────────────────────────────────────

    def _make_logging_tqdm(self, base_cls, log, lock):
        """Build a tqdm subclass that writes progress to our log list."""
        class _LoggingTqdm(base_cls):
            def display(self, msg=None, pos=None):
                line = _strip_ansi(self.__str__()).strip()
                if line:
                    with lock:
                        # Overwrite last progress line if it exists
                        if log and log[-1].startswith("[progress]"):
                            log[-1] = f"[progress] {line}"
                        else:
                            log.append(f"[progress] {line}")

            def close(self):
                self.display()
                try:
                    super().close()
                except Exception:
                    pass

        return _LoggingTqdm

    def _patch_tqdm(self):
        log = self._lines
        lock = self._lock

        # All the places tqdm lives that HuggingFace and transformers import from
        targets = [
            ("tqdm", "tqdm"),
            ("tqdm.auto", "tqdm"),
            ("tqdm.std", "tqdm"),
            ("tqdm.notebook", "tqdm"),
            ("huggingface_hub.utils._tqdm", "tqdm"),
            ("huggingface_hub.utils", "tqdm"),
        ]

        for mod_path, attr in targets:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                orig = getattr(mod, attr, None)
                if orig is None:
                    continue
                patched = self._make_logging_tqdm(orig, log, lock)
                setattr(mod, attr, patched)
                self._saved[f"{mod_path}.{attr}"] = (mod, attr, orig)
            except Exception:
                pass

    def _unpatch_tqdm(self):
        for key, (mod, attr, orig) in self._saved.items():
            try:
                setattr(mod, attr, orig)
            except Exception:
                pass
        self._saved.clear()
