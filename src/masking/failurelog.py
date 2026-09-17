import csv
from datetime import datetime
from pathlib import Path
class FailureLog:
    """
    calss to make a .csv file for all of the failure mediapipe detection
    """
    def __init__(self, log_path, flush_every: int = 100):
        self.log_path = Path(log_path)
        self.flush_every = flush_every
        self._buffer = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not self.log_path.exists() or self.log_path.stat().st_size == 0
        self._file = open(self.log_path, mode='a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        if is_new_file:
            self._writer.writerow(["image_name", "split", "variant", "reason", "timestamp"])

    def log_failure(self, image_name, split, variant, reason="no_face_detected"):
        self._buffer.append([
            image_name, split, variant, reason,
            datetime.now().isoformat(timespec='seconds'),
        ])
        if len(self._buffer) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()

    def close(self):
        self._flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
