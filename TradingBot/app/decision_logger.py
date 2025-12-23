import os
import glob
import json
import threading
import queue
import time
import atexit
from typing import Optional, List

from .config import DECISION_LOG_MAX_MB, DECISION_LOG_MAX_FILES


class RotatingDecisionLogger:
    """
    Background writer that persists decision logs to size-limited rotating files.
    Keeps logging non-blocking for the trading loop by delegating all disk IO to a worker thread.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        file_prefix: str = "decisions_",
        max_bytes: Optional[int] = None,
        max_files: Optional[int] = None,
    ):
        self.log_dir = log_dir
        self.file_prefix = file_prefix
        self.max_bytes = int(max_bytes or (DECISION_LOG_MAX_MB * 1024 * 1024))
        self.max_files = int(max_files or DECISION_LOG_MAX_FILES)

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._worker, name="DecisionLogWriter", daemon=True)

        self._current_file = None  # type: Optional[object]
        self._current_path: Optional[str] = None
        self._current_size: int = 0
        self._current_index: int = 0
        self._file_lock = threading.Lock()

        self._prepare_log_dir()
        self._initialize_file_handle()
        self._writer_thread.start()

    def log(self, entry: dict) -> None:
        """Queue a log entry for asynchronous persistence."""
        if self._stop_event.is_set():
            return

        try:
            payload = json.dumps(entry)
        except Exception:
            return  # Skip malformed entries silently (do not impact trading loop)

        try:
            self._queue.put_nowait(payload + "\n")
        except queue.Full:
            # Shouldn't occur with infinite queue, but guard just in case
            pass

    def shutdown(self, timeout: float = 2.0) -> None:
        """Flush pending logs and stop the background thread."""
        self._stop_event.set()
        self._queue.put_nowait(None)  # Sentinel to unblock worker
        self._writer_thread.join(timeout=timeout)
        with self._file_lock:
            if self._current_file:
                try:
                    self._current_file.flush()
                except Exception:
                    pass
                try:
                    self._current_file.close()
                except Exception:
                    pass
                self._current_file = None

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────
    def _prepare_log_dir(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

        # Rename legacy decisions.jsonl (if present) to keep the directory tidy.
        legacy_path = os.path.join(self.log_dir, "decisions.jsonl")
        if os.path.exists(legacy_path):
            legacy_target = os.path.join(
                self.log_dir,
                f"{self.file_prefix}legacy_{int(time.time())}.jsonl",
            )
            try:
                os.replace(legacy_path, legacy_target)
            except OSError:
                pass

    def _initialize_file_handle(self) -> None:
        existing_files = self._list_log_files()
        if existing_files:
            latest_path = existing_files[-1]
            latest_index = self._extract_index(latest_path)
            size = os.path.getsize(latest_path)
            if size < self.max_bytes:
                self._open_file(latest_path, latest_index, size)
                return
            else:
                # Need to start with a new file after the latest
                self._open_new_file(latest_index + 1)
                return
        # No existing logs -> start at index 1
        self._open_new_file(1)

    def _list_log_files(self) -> List[str]:
        pattern = os.path.join(self.log_dir, f"{self.file_prefix}[0-9][0-9][0-9][0-9].jsonl")
        files = glob.glob(pattern)
        files_with_index = []
        for path in files:
            idx = self._extract_index(path)
            if idx is not None:
                files_with_index.append((idx, path))
        files_with_index.sort(key=lambda x: x[0])
        return [path for _, path in files_with_index]

    def _extract_index(self, path: str) -> Optional[int]:
        filename = os.path.basename(path)
        try:
            numeric = filename.replace(self.file_prefix, "").replace(".jsonl", "")
            return int(numeric)
        except ValueError:
            return None

    def _open_file(self, path: str, index: int, size: int) -> None:
        self._current_path = path
        self._current_index = index
        self._current_size = size
        self._current_file = open(path, "a", encoding="utf-8", buffering=1)

    def _open_new_file(self, index: int) -> None:
        path = os.path.join(self.log_dir, f"{self.file_prefix}{index:04d}.jsonl")
        self._current_path = path
        self._current_index = index
        self._current_size = os.path.getsize(path) if os.path.exists(path) else 0
        self._current_file = open(path, "a", encoding="utf-8", buffering=1)
        self._enforce_retention()

    def _rotate_file(self) -> None:
        if self._current_file:
            try:
                self._current_file.flush()
            except Exception:
                pass
            try:
                self._current_file.close()
            except Exception:
                pass
        next_index = self._current_index + 1 if self._current_index else 1
        self._open_new_file(next_index)

    def _enforce_retention(self) -> None:
        if self.max_files <= 0:
            return
        files = self._list_log_files()
        if len(files) <= self.max_files:
            return
        excess = len(files) - self.max_files
        for path in files[:excess]:
            try:
                os.remove(path)
            except OSError:
                pass

    def _worker(self) -> None:
        while True:
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if payload is None:
                # Sentinel for shutdown
                self._queue.task_done()
                break

            try:
                self._write_payload(payload)
            finally:
                self._queue.task_done()

        # Flush remaining items if any
        while not self._queue.empty():
            payload = self._queue.get_nowait()
            if payload:
                self._write_payload(payload)
            self._queue.task_done()

    def _write_payload(self, payload: str) -> None:
        data_size = len(payload.encode("utf-8"))
        with self._file_lock:
            if (self._current_file is None) or (self._current_size + data_size > self.max_bytes):
                self._rotate_file()

            if self._current_file:
                try:
                    self._current_file.write(payload)
                    self._current_size += data_size
                except Exception:
                    # If write fails, try rotating once to recover
                    try:
                        self._rotate_file()
                        self._current_file.write(payload)
                        self._current_size += data_size
                    except Exception:
                        pass


_singleton_logger: Optional[RotatingDecisionLogger] = None
_singleton_lock = threading.Lock()


def get_decision_logger() -> RotatingDecisionLogger:
    global _singleton_logger
    if _singleton_logger is None:
        with _singleton_lock:
            if _singleton_logger is None:
                _singleton_logger = RotatingDecisionLogger()
                atexit.register(_singleton_logger.shutdown)
    return _singleton_logger

