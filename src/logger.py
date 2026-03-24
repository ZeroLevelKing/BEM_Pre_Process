import logging
import os
import gmsh
import time

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging():
    log_dir = os.path.join("out", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "process.log")

    # Ensure standard handlers
    handler = FlushFileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)


def monitor_gmsh_logs(stop_event):
    while not stop_event.is_set():
        try:
            logs = gmsh.logger.get()
            for msg in logs:
                logging.info(f"Gmsh: {msg}")
        except Exception:
            pass
        time.sleep(0.5)
