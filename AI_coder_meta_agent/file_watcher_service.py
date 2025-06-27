import logging
import os
import threading
import time

import requests
from flask import Flask, jsonify
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CHANGED: Use PROJECT_ROOT from environment variable (Docker compatible)
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/project")
logger.info(f"File Watcher Service: Monitoring project root: {PROJECT_ROOT}")

# CHANGED: Use Docker service name for orchestration URL
ORCHESTRATION_SERVICE_URL = os.getenv(
    "ORCHESTRATION_SERVICE_URL", "http://orchestration_service:5006"
)
ORCHESTRATION_NOTIFY_ENDPOINT = f"{ORCHESTRATION_SERVICE_URL}/notify/file_change"
logger.info(f"Orchestration endpoint: {ORCHESTRATION_NOTIFY_ENDPOINT}")


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, observer_instance):
        self.observer = observer_instance
        self.debounce_times = {}
        self.DEBOUNCE_PERIOD = 0.5  # seconds

    def _send_notification(self, event_type, path):
        current_time = time.time()
        if path in self.debounce_times and (
            current_time - self.debounce_times[path] < self.DEBOUNCE_PERIOD
        ):
            return

        self.debounce_times[path] = current_time

        try:
            relative_path = os.path.relpath(path, PROJECT_ROOT)
        except ValueError:
            logger.warning(f"Ignoring event for path outside project root: {path}")
            return

        # --- MODIFIED FILTER LOGIC HERE TO EXCLUDE ARCHIVE AND ALLOW .json ---
        is_design_card_dir = relative_path.startswith("design_cards/")
        is_archive_path = (
            "design_cards/archive/" in relative_path
        )  # Checks for '/archive/' anywhere under design_cards/
        is_md_or_json_file = relative_path.endswith(".md") or relative_path.endswith(
            ".json"
        )

        # Only proceed if it's within design_cards/, NOT in archive, and is a .md or .json file
        if not (is_design_card_dir and not is_archive_path and is_md_or_json_file):
            logger.debug(
                f"Ignoring non-active design card file: {relative_path} ({event_type})"
            )
            return
        # --- END MODIFIED FILTER ---

        notification_payload = {
            "event_type": event_type,
            "file_path": relative_path,
            "timestamp": current_time,
        }

        try:
            response = requests.post(
                ORCHESTRATION_NOTIFY_ENDPOINT, json=notification_payload, timeout=120
            )
            response.raise_for_status()
            logger.info(f"Notified Orchestrator about {event_type} on {relative_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to notify Orchestrator: {e}", exc_info=True)

    def on_created(self, event):
        if not event.is_directory:
            self._send_notification("created", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._send_notification("modified", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._send_notification("deleted", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # When a file is moved, it triggers a 'deleted' for src and 'created' for dest
            # We explicitly handle it here to ensure correct filtering.
            self._send_notification("deleted", event.src_path)
            self._send_notification("created", event.dest_path)


# Initialize observer
observer = Observer()
event_handler = ChangeHandler(observer)
watch_path = PROJECT_ROOT
observer.schedule(event_handler, watch_path, recursive=True)
logger.info(f"Scheduled observer for path: {watch_path}")


def start_watcher():
    observer.start()
    logger.info("Observer started")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logger.info("Observer stopped")


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "running", "watching": watch_path}), 200


if __name__ == "__main__":
    PORT = int(os.getenv("FILE_WATCHER_SERVICE_PORT", 5003))
    logger.info(f"Starting File Watcher Service on port {PORT}")

    watcher_thread = threading.Thread(target=start_watcher)
    watcher_thread.daemon = True
    watcher_thread.start()

    app.run(host="0.0.0.0", port=PORT)
