# AI_coder_meta_agent/orchestration_service.py

import asyncio
import json
import logging
import os
import re
import threading
import time
from logging.handlers import RotatingFileHandler
import shutil # Added for archiving

import httpx
import uvicorn
from flask import Flask, jsonify, request
from uvicorn.middleware.wsgi import WSGIMiddleware

# --- Differentiated Logging Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_log_level_str = os.getenv("LOG_LEVEL_CONSOLE", "INFO").upper()
console_handler.setLevel(getattr(logging, console_log_level_str))
console_formatter = logging.Formatter(
    "%(name)s | %(asctime)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, "ai_coder_full.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_log_level_str = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()
file_handler.setLevel(getattr(logging, file_log_level_str))
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s | %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logging.getLogger("httpx").setLevel(os.getenv("LOG_LEVEL_HTTPX", "WARNING").upper())
logging.getLogger("urllib3").setLevel(os.getenv("LOG_LEVEL_URLLIB3", "WARNING").upper())
logging.getLogger("uvicorn").setLevel(os.getenv("LOG_LEVEL_UVICORN", "INFO").upper())
logging.getLogger("uvicorn.access").setLevel(
    os.getenv("LOG_LEVEL_UVICORN_ACCESS", "INFO").upper()
)
logging.getLogger("uvicorn.error").setLevel(
    os.getenv("LOG_LEVEL_UVICORN_ERROR", "WARNING").upper()
)

logger = logging.getLogger(__name__)
logger.info("Differentiated logging configured for Orchestration Service.")
# --- End Differentiated Logging Setup ---


class OrchestrationService:
    STATE_FILE = "orchestrator_state.json"

    def __init__(self):
        self.LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:5001")
        self.FILE_SYSTEM_SERVICE_URL = os.getenv(
            "FILE_SYSTEM_SERVICE_URL", "http://localhost:5002"
        )
        self.TEST_EXECUTION_SERVICE_URL = os.getenv(
            "TEST_EXECUTION_SERVICE_URL", "http://localhost:5004"
        )
        self.MAX_TASK_ITERATIONS = int(os.getenv("MAX_TASK_ITERATIONS", 10))

        self.PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/project")

        # Added: Configurable design cards directory
        self.DESIGN_CARDS_DIR = os.getenv("DESIGN_CARDS_DIR", "design_cards")
        self.COMPLETED_CARDS_DIR = os.getenv("COMPLETED_CARDS_DIR", "completed_design_cards")

        # Ensure directories exist for archiving
        os.makedirs(os.path.join(self.PROJECT_ROOT, self.COMPLETED_CARDS_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.PROJECT_ROOT, self.COMPLETED_CARDS_DIR, "archive"), exist_ok=True)


        # In-memory queue for design cards - NOW asyncio.Queue
        self.task_queue = asyncio.Queue()
        # Dictionary to store active task contexts, keyed by design card path
        self.active_tasks = {}
        # Thread-safe lock for accessing active_tasks
        self.lock = threading.Lock()

        # Variable to hold the main asyncio event loop instance (for thread-safe coroutine submission)
        self.main_asyncio_loop = None

        logger.info("Orchestration Service initialized.")

    def _get_abs_path(self, relative_path):
        return os.path.join(self.PROJECT_ROOT, relative_path)

    def _load_state(self):
        """Loads the active_tasks state from a JSON file."""
        try:
            with open(self.STATE_FILE, "r") as f:
                state_data = json.load(f)
                # Only load tasks that are not complete, as they need further processing
                # We also clean up the state here to remove completed/failed tasks from in-memory active_tasks
                # The physical file will be archived.
                self.active_tasks = {
                    k: v for k, v in state_data.items()
                    if v["status"] not in ["TASK_COMPLETE", "TASK_FAILED"]
                }
                logger.info(f"Loaded {len(state_data)} tasks from state. {len(self.active_tasks)} are active.")
        except FileNotFoundError:
            logger.info("No state file found. Starting empty.")
            self.active_tasks = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding state file: {e}. Starting empty.", exc_info=True)
            self.active_tasks = {}
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}. Starting empty.", exc_info=True)
            self.active_tasks = {}

    def _save_state_sync(self):
        """Saves the active_tasks state to a JSON file (synchronously)."""
        # This is called from the async processing loop via asyncio.to_thread
        try:
            # Reintroducing compression for large fields before saving state
            persisted_tasks = {}
            with self.lock: # Acquire lock before iterating and modifying context for saving
                for path, context in self.active_tasks.items():
                    compressed_context = context.copy()
                    # Remove large, transient data fields that are re-read as needed
                    compressed_context.pop("design_card_content", None)
                    # Decide if previous_llm_responses should be compressed or truncated if it gets too large
                    # For now, keeping it as is, but it's a known potential bloat factor.
                    persisted_tasks[path] = compressed_context

            with open(self.STATE_FILE, "w") as f:
                json.dump(persisted_tasks, f, indent=4) # Save compressed tasks
            logger.debug("State saved successfully.")
        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)

    async def _initial_load_and_queue_tasks(self):
        """
        Loads existing tasks from the persistent state file and re-queues
        tasks that are still IN_PROGRESS. Also performs cleanup for orphaned state entries.
        """
        self._load_state() # Load state synchronously
        with self.lock:
            for path, task_context in list(self.active_tasks.items()): # Iterate over copy
                if task_context.get("status") in ["PENDING", "IN_PROGRESS"]:
                    logger.info(
                        f"Re-queuing IN_PROGRESS/PENDING task: {os.path.basename(path)}"
                    )
                    await self.task_queue.put(path)
                # Removed specific handling for TASK_COMPLETE/TASK_FAILED here,
                # as _load_state already filters them out from self.active_tasks.
                # Archiving now handles physical file movement.
        await asyncio.to_thread(self._save_state_sync) # Save state after potential re-queueing (if active_tasks changed)


    async def _call_llm_communication(
        self, task_type: str, input_data: dict = None, custom_prompt: str = None
    ):
        """Makes an asynchronous HTTP call to the LLM Communication Service."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.LLM_SERVICE_URL}/llm/request_code",
                    json={
                        "task_type": task_type,
                        "input_data": input_data,
                        "custom_prompt": custom_prompt,
                    },
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from {self.LLM_SERVICE_URL}/llm/request_code: {e.response.text}",
                exc_info=True,
            )
            return {"status": "error", "message": f"HTTP Error: {e.response.text}"}
        except httpx.RequestError as e:
            logger.error(
                f"Network or request error communicating with LLM Service: {e}",
                exc_info=True,
            )
            return {"status": "error", "message": f"Network Error: {e}"}
        except Exception as e:
            logger.error(
                f"Unexpected error calling LLM Communication Service: {e}",
                exc_info=True,
            )
            return {"status": "error", "message": f"Unexpected Error: {e}"}

    async def _read_file_content(self, file_path: str) -> str:
        """Reads file content from the File System Service."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.FILE_SYSTEM_SERVICE_URL}/files/read",
                    json={"file_path": file_path},
                )
                response.raise_for_status()
                return response.json().get("content", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404: # File not found is a specific case, return empty
                logger.warning(f"File not found on FS Service: {file_path}")
                return ""
            logger.error(
                f"HTTP error from {self.FILE_SYSTEM_SERVICE_URL}/files/read: {e.response.text}",
                exc_info=True,
            )
            return ""
        except httpx.RequestError as e:
            logger.error(
                f"Network error reading file from File System Service: {e}",
                exc_info=True,
            )
            return ""

    async def _write_file_content(self, file_path: str, content: str) -> bool:
        """Writes content to a file via the File System Service."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.FILE_SYSTEM_SERVICE_URL}/files/write",
                    json={"file_path": file_path, "content": content},
                )
                response.raise_for_status()
                logger.info(f"Successfully wrote to {file_path}")
                return True
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error writing to {file_path}: {e.response.text}", exc_info=True
            )
            return False
        except httpx.RequestError as e:
            logger.error(f"Network error writing to {file_path}: {e}", exc_info=True)
            return False

    async def _delete_file_fs(self, file_path: str) -> bool:
        """Deletes a file via the File System Service."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.FILE_SYSTEM_SERVICE_URL}/files/delete",
                    json={"file_path": file_path},
                )
                response.raise_for_status()
                logger.info(f"Successfully deleted {file_path} via FS Service.")
                return True
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error deleting {file_path} via FS Service: {e.response.text}", exc_info=True
            )
            return False
        except httpx.RequestError as e:
            logger.error(f"Network error deleting {file_path} via FS Service: {e}", exc_info=True)
            return False

    async def _run_tests(
        self, test_file_path: str, app_file_path: str
    ) -> dict:
        """Runs tests via the Test Execution Service."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.TEST_EXECUTION_SERVICE_URL}/tests/run",
                    json={
                        "test_context": test_file_path,
                        "app_context": app_file_path,
                    },
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from Test Execution Service: {e.response.text}",
                exc_info=True,
            )
            return {"status": "error", "message": f"HTTP Error: {e.response.text}"}
        except httpx.RequestError as e:
            logger.error(
                f"Network error communicating with Test Execution Service: {e}",
                exc_info=True,
            )
            return {"status": "error", "message": f"Network Error: {e}"}

    async def process_queued_tasks(self):
        """Asynchronous loop to process tasks from the queue."""
        logger.info("Starting task queue processing loop.")
        while True:
            design_card_path = await self.task_queue.get()
            logger.debug(f"Retrieved task from queue: {design_card_path}")

            with self.lock:
                task_context = self.active_tasks.get(design_card_path)

            # --- Task Deduplication/Status Check ---
            # Added: Simple deduplication/status check to avoid redundant processing
            # If a task is already IN_PROGRESS and recently modified, it might be re-queued.
            # For robust deduplication, a more complex mechanism (e.g., tracking hash of content)
            # would be needed, but this handles simple cases.
            if task_context is None:
                logger.info(f"Task {design_card_path} no longer in active_tasks or was removed. Skipping.")
                self.task_queue.task_done()
                continue
            elif task_context["status"] in ["TASK_COMPLETE", "TASK_FAILED"]:
                logger.info(
                    f"Task {design_card_path} is already {task_context['status']}. Not re-processing."
                )
                self.task_queue.task_done()
                continue
            elif task_context["status"] == "IN_PROGRESS" and task_context.get("processing_lock_active", False):
                # This is a very basic attempt to avoid processing the exact same task
                # if it was re-queued very quickly while already being processed.
                # A proper solution would involve a per-task lock or a hash of content.
                logger.debug(f"Task {design_card_path} already marked as 'processing_lock_active'. Skipping this queue entry.")
                self.task_queue.task_done()
                continue

            with self.lock:
                task_context["processing_lock_active"] = True # Mark as being processed

            logger.info(f"Processing design card: {design_card_path}")
            try:
                await self._process_design_card_iteration(design_card_path)
            except Exception as e:
                logger.error(
                    f"Critical error processing {design_card_path}: {e}",
                    exc_info=True,
                )
                with self.lock:
                    task_context["status"] = "TASK_FAILED"
                    task_context["final_feedback"] = f"Critical internal error: {e}"
            finally:
                # Always save state after an iteration attempt
                await asyncio.to_thread(self._save_state_sync)

                with self.lock:
                    # Remove processing lock
                    task_context["processing_lock_active"] = False

                    # If task is completed or failed, remove from active_tasks as it's being archived
                    if task_context["status"] in ["TASK_COMPLETE", "TASK_FAILED"]:
                        del self.active_tasks[design_card_path]
                        # Archiving handled by _process_design_card_iteration directly now

                self.task_queue.task_done()

                # Re-queue if not complete or failed
                with self.lock: # Re-acquire lock for final status check and potential re-queueing
                    if task_context["status"] not in ["TASK_COMPLETE", "TASK_FAILED"] and design_card_path in self.active_tasks:
                        # Only re-queue if it's still an active task (not deleted/archived during processing)
                        await self.task_queue.put(design_card_path)
                        logger.debug(f"Re-queued {design_card_path} for next iteration.")


    async def _process_design_card_iteration(self, design_card_path: str):
        """
        Executes a single iteration of the development cycle for a design card.
        """
        with self.lock:
            task_context = self.active_tasks[design_card_path]

        current_iteration = task_context["iteration_count"] + 1
        task_context["iteration_count"] = current_iteration

        if current_iteration > self.MAX_TASK_ITERATIONS:
            logger.error(f"Task failed: Exceeded max iterations ({self.MAX_TASK_ITERATIONS}) for {os.path.basename(design_card_path)}.")
            task_context["status"] = "TASK_FAILED"
            task_context["final_feedback"] = "Exceeded max iterations"
            await self._archive_design_card(design_card_path) # Archive failed tasks
            return

        logger.info(
            f"\n--- Strategic Planning Phase (Iteration {current_iteration}/{self.MAX_TASK_ITERATIONS}) ---"
        )

        planner_decision = {}
        next_task_type = ""
        generated_prompt = ""
        generated_sub_design_card_json = None
        debug_target_file = None
        reasoning = ""
        llm_planner_called = False

        current_app_code = ""
        current_test_code = ""
        if task_context["app_file_path"]:
            current_app_code = await self._read_file_content(task_context["app_file_path"])
        if task_context["test_file_path"]:
            current_test_code = await self._read_file_content(task_context["test_file_path"])
        logger.debug(f"Current app code length: {len(current_app_code)}. Current test code length: {len(current_test_code)}")

        if (
            current_iteration == 1
            and not current_app_code
            and not current_test_code
        ):
            logger.info("Hardcoding Planner decision for initial RED state: Generate Tests.")
            next_task_type = "TEST_GENERATION"
            generated_prompt = (
                f"Design Card: {task_context['design_card_content']}\n\n"
                f"Current Application Code: (None)\n\n"
                f"Current Test Code: (None)\n\n"
                f"Test Results: (None)\n\n"
                f"Code Review Feedback: (None)\n\n"
                f"Write comprehensive unit tests for the functionality described in the design card. "
                f"The application code does not exist yet, so generate tests that will initially fail (RED state). "
                f"Output ONLY the Python test code within a markdown block (```python\\n[YOUR TEST CODE HERE]\\n```)."
            )
            reasoning = "Hardcoded decision for initial RED state: No app/test code, generate tests first."
            task_context["last_planner_decision_type"] = next_task_type
            task_context["status"] = "IN_PROGRESS"
        else:
            planner_input_data = {
                "design_card_content": task_context["design_card_content"],
                "current_app_code": current_app_code,
                "current_test_code": current_test_code,
                "test_results": task_context["last_test_results"],
                "code_review_feedback": task_context["last_code_review_feedback"],
                "iteration_count": current_iteration,
                "previous_llm_responses": task_context["previous_llm_responses"],
                "available_templates": [
                    "TEST_GENERATION", "CODE_GENERATION", "DEBUG_CODE_GENERATION",
                    "CODE_REVIEW", "CODE_REVIEW_DEBUG_CONTEXT",
                    "GENERATE_SUB_DESIGN_CARD", "REFACTOR_SUB_DESIGN_CARD", "REFACTOR_CODE"
                ],
                "current_system_architecture_overview": ""
            }
            logger.debug(f"Calling Planner LLM for task: {design_card_path}")
            planner_response = await self._call_llm_communication(
                task_type="PLANNING", input_data=planner_input_data
            )
            llm_planner_called = True

            if planner_response.get("status") == "error":
                logger.error(f"LLM Communication Service reported a critical error for Planner: {planner_response.get('message')}. Failing task.")
                task_context["status"] = "TASK_FAILED"
                task_context["final_feedback"] = planner_response.get("message")
                await self._archive_design_card(design_card_path) # Archive failed tasks
                return

            planner_decision = planner_response.get("generated_code")

            if not isinstance(planner_decision, dict):
                logger.error(f"Planner response was not a dict after normalization. Raw: {planner_decision}. Failing task.")
                task_context["status"] = "TASK_FAILED"
                task_context["final_feedback"] = "Planner returned unparseable output (unexpected format after normalization)."
                await self._archive_design_card(design_card_path) # Archive failed tasks
                return

            next_task_type = planner_decision.get("next_llm_task_type")
            generated_prompt = planner_decision.get("generated_prompt")
            generated_sub_design_card_json = planner_decision.get("generated_sub_design_card_json")
            debug_target_file = planner_decision.get("debug_target_file")
            reasoning = planner_decision.get("reasoning", "No specific reasoning provided.")

            logger.info(f"Planner decided: {next_task_type}")
            logger.debug(f"Planner Reasoning: {reasoning}")
            logger.debug(f"Generated prompt (partial): {generated_prompt[:200]}...")

            task_context["last_planner_decision_type"] = next_task_type
            task_context["status"] = "IN_PROGRESS"

        with self.lock:
            task_context["previous_llm_responses"].append({
                "iteration": current_iteration,
                "task_executed": "PLANNING",
                "planner_decision": planner_decision
            })
        await asyncio.to_thread(self._save_state_sync)


        llm_response = {"status": "error", "generated_code": "", "message": "Task not executed by LLM."}
        target_file = None
        llm_task_message = f"Executing {next_task_type} for iteration {current_iteration}."

        if next_task_type == "TASK_COMPLETE":
            with self.lock:
                task_context["status"] = "TASK_COMPLETE"
                task_context["final_feedback"] = reasoning
            logger.info(f"Task {design_card_path} marked as COMPLETE.")
            await self._archive_design_card(design_card_path) # Archive completed tasks
            return

        elif next_task_type == "TASK_FAILED":
            with self.lock:
                task_context["status"] = "TASK_FAILED"
                task_context["final_feedback"] = reasoning
            logger.info(f"Task {design_card_path} marked as FAILED.")
            await self._archive_design_card(design_card_path) # Archive failed tasks
            return

        elif next_task_type == "GENERATE_SUB_DESIGN_CARD":
            if generated_sub_design_card_json:
                for sub_card_data in generated_sub_design_card_json.get("sub_design_cards", []):
                    sub_card_name = sub_card_data.get("name", f"sub_card_{time.time()}.json")
                    # Relative path for sub-cards within the design_cards directory
                    sub_card_relative_path = os.path.join(self.DESIGN_CARDS_DIR, "sub_cards", sub_card_name.replace(' ', '_').lower() + ".json")

                    # Add new sub-design card to active tasks. This method will also queue it.
                    await self.add_design_card(sub_card_relative_path, sub_card_data)

                logger.info(f"Generated sub-design cards for {design_card_path}. Parent task status remains IN_PROGRESS for now.")
                with self.lock:
                    task_context["status"] = "IN_PROGRESS"
            else:
                logger.error(f"Planner decided GENERATE_SUB_DESIGN_CARD but no generated_sub_design_card_json provided.")
                with self.lock:
                    task_context["status"] = "TASK_FAILED"
                    task_context["final_feedback"] = "Planner failed to provide sub-design card JSON."
                await self._archive_design_card(design_card_path) # Archive failed tasks
            return

        else:
            logger.info(llm_task_message)
            llm_response = await self._call_llm_communication(
                task_type=next_task_type, custom_prompt=generated_prompt, input_data={}
            )

            if llm_response.get("status") == "error":
                logger.error(f"LLM Communication Service reported error for {next_task_type}: {llm_response.get('message')}. Failing task.")
                with self.lock:
                    task_context["status"] = "TASK_FAILED"
                    task_context["final_feedback"] = llm_response.get("message")
                await self._archive_design_card(design_card_path) # Archive failed tasks
                return

        llm_response_content = llm_response.get("generated_code", "")

        with self.lock:
            task_context["previous_llm_responses"].append({
                "iteration": current_iteration,
                "task_executed": next_task_type,
                "response": llm_response
            })
        await asyncio.to_thread(self._save_state_sync)

        if next_task_type in [
            "CODE_GENERATION",
            "TEST_GENERATION",
            "DEBUG_CODE_GENERATION",
            "REFACTOR_CODE",
        ]:
            if next_task_type == "TEST_GENERATION" or debug_target_file == "test_file":
                target_file = task_context["test_file_path"]
            elif next_task_type == "CODE_GENERATION" or debug_target_file == "app_file":
                target_file = task_context["app_file_path"]
            elif next_task_type == "REFACTOR_CODE":
                target_file = task_context["app_file_path"]

            if target_file:
                logger.debug(f"Attempting to write {len(llm_response_content)} bytes to file: {target_file}. Preview: {llm_response_content[:100]}...")
                success = await self._write_file_content(target_file, llm_response_content)
                if not success:
                    with self.lock:
                        task_context["status"] = "TASK_FAILED"
                        task_context["final_feedback"] = f"Failed to write generated code to {target_file}."
                    await self._archive_design_card(design_card_path) # Archive failed tasks
                    return

                logger.info(f"Running tests for {task_context['test_file_path']} against {task_context['app_file_path']}")
                test_results = await self._run_tests(task_context["test_file_path"], task_context["app_file_path"])
                with self.lock:
                    task_context["last_test_results"] = test_results
                    task_context["status"] = "IN_PROGRESS"
                await asyncio.to_thread(self._save_state_sync)

            else:
                logger.error(f"No target file path determined for {next_task_type}.")
                with self.lock:
                    task_context["status"] = "TASK_FAILED"
                    task_context["final_feedback"] = f"No target file path for {next_task_type}."
                await self._archive_design_card(design_card_path) # Archive failed tasks
                return

        if next_task_type in ["CODE_REVIEW", "CODE_REVIEW_DEBUG_CONTEXT"]:
            try:
                feedback = json.loads(llm_response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in code review feedback: {e}. Raw content: {llm_response_content[:500]}...")
                feedback = {"status": "ERROR", "summary": "Invalid JSON format from code review LLM."}

            with self.lock:
                task_context["last_code_review_feedback"] = feedback
                if not isinstance(task_context["last_code_review_feedback"], dict):
                    logger.error(f"Code Review feedback was not a dict after normalization. Raw: {task_context['last_code_review_feedback']}. Failing task.")
                    task_context["status"] = "TASK_FAILED"
                    task_context["final_feedback"] = "Code Review LLM returned unparseable output after normalization."
                    await self._archive_design_card(design_card_path) # Archive failed tasks
                    return

                if task_context["last_code_review_feedback"].get("status") == "PASSED":
                    logger.info("Code Review Passed.")
                    if task_context["last_test_results"].get("overall_result") == "Pass":
                        task_context["status"] = "TASK_COMPLETE"
                        task_context["final_feedback"] = "Code review passed and tests are passing."
                        logger.info(f"Task {design_card_path} marked as COMPLETE due to passing tests and review.")
                        await self._archive_design_card(design_card_path) # Archive completed tasks
                    else:
                        logger.info("Code Review Passed, but tests are still failing. Re-queueing for further development.")
                        task_context["status"] = "IN_PROGRESS"
                else:
                    logger.warning("Code Review FAILED. Re-queueing for further development.")
                    task_context["status"] = "IN_PROGRESS"
            await asyncio.to_thread(self._save_state_sync)

    async def _archive_design_card(self, design_card_path: str):
        """Archives a completed or failed design card by moving it to a designated directory."""
        if not design_card_path:
            logger.warning("Attempted to archive with an empty design_card_path.")
            return

        # Ensure design_card_path is an absolute path within PROJECT_ROOT
        abs_design_card_path = self._get_abs_path(design_card_path)

        # Determine if it's a sub-design card
        is_sub_card = self.DESIGN_CARDS_DIR + "/sub_cards/" in abs_design_card_path

        # Determine target archive directory
        relative_archive_dir = os.path.join(self.COMPLETED_CARDS_DIR, "archive")
        if is_sub_card:
            relative_archive_dir = os.path.join(relative_archive_dir, "sub_cards")

        target_archive_dir = self._get_abs_path(relative_archive_dir)
        os.makedirs(target_archive_dir, exist_ok=True) # Ensure target archive directory exists

        # Construct new path in archive
        archive_file_name = os.path.basename(abs_design_card_path)
        archive_path = os.path.join(target_archive_dir, archive_file_name)

        logger.info(f"Attempting to archive {abs_design_card_path} to {archive_path}")

        try:
            # Check if source file exists before attempting to move
            # Note: We are now operating on local paths for shutil.move, assuming FS service manages /project
            # For a true microservices approach, moving/deleting files would still go via FS service.
            # For now, this assumes the orchestrator can access the underlying file system within its container.
            if os.path.exists(abs_design_card_path):
                shutil.move(abs_design_card_path, archive_path)
                logger.info(f"Successfully archived design card: {design_card_path}")
            else:
                logger.warning(f"Design card file not found for archiving: {design_card_path}. It might have been deleted already.")

            # Additionally, remove the task from active_tasks if it's still there
            with self.lock:
                if design_card_path in self.active_tasks:
                    del self.active_tasks[design_card_path]
            await asyncio.to_thread(self._save_state_sync) # Save state after archiving and removal
        except Exception as e:
            logger.error(f"Failed to archive design card {design_card_path}: {e}", exc_info=True)


    async def add_design_card(self, file_path: str, content: str | dict):
        """Adds a new design card to the active tasks and queues it for processing."""
        # Note: file_path here is expected to be relative to PROJECT_ROOT,
        # e.g., "design_cards/new_feature.md" or "design_cards/sub_cards/login_flow.json"

        abs_file_path = self._get_abs_path(file_path)

        if isinstance(content, dict):
            content_to_write = json.dumps(content, indent=2)
            design_card_content_for_context = content_to_write
        else:
            content_to_write = content
            design_card_content_for_context = content

        success = await self._write_file_content(abs_file_path, content_to_write)
        if not success:
            logger.error(f"Failed to write new design card file: {abs_file_path}. Not adding to queue.")
            return

        app_file_path = ""
        test_file_path = ""
        target_app_module_name = ""
        target_test_file_basename = ""

        try:
            parsed_content = json.loads(design_card_content_for_context)
            if "target_files" in parsed_content:
                app_file_path = parsed_content["target_files"].get("app_module_path", "")
                test_file_path = parsed_content["target_files"].get("test_module_path", "")
                target_app_module_name = parsed_content["target_files"].get("app_module_name", "")
                target_test_file_basename = parsed_content["target_files"].get("test_module_name", "")
            else:
                base_name = parsed_content.get(
                    "id", os.path.splitext(os.path.basename(abs_file_path))[0]
                )
                app_file_path = os.path.join(self.PROJECT_ROOT, "src", f"{base_name}.py")
                test_file_path = os.path.join(self.PROJECT_ROOT, "tests", f"test_{base_name}.py")
                target_app_module_name = base_name
                target_test_file_basename = f"test_{base_name}"
        except json.JSONDecodeError:
            base_name = (
                os.path.splitext(os.path.basename(abs_file_path))[0]
                .replace("test_", "")
                .lower()
                .replace("-", "_")
            )
            base_name = re.sub(r"[^a-z0-9_]", "", base_name)
            app_file_path = os.path.join(self.PROJECT_ROOT, "src", f"{base_name}.py")
            test_file_path = os.path.join(self.PROJECT_ROOT, "tests", f"test_{base_name}.py")
            target_app_module_name = base_name
            target_test_file_basename = f"test_{base_name}"


        with self.lock:
            self.active_tasks[abs_file_path] = {
                "design_card_path": abs_file_path,
                "design_card_content": design_card_content_for_context,
                "app_file_path": app_file_path,
                "test_file_path": test_file_path,
                "target_app_module_name": target_app_module_name,
                "target_test_file_basename": target_test_file_basename,
                "iteration_count": 0,
                "previous_llm_responses": [],
                "last_test_results": {"status": "NOT_STARTED", "output": ""},
                "last_code_review_feedback": {"status": "NOT_STARTED", "summary": ""},
                "status": "PENDING",
                "final_feedback": "",
                "last_planner_decision_type": "INIT",
                "processing_lock_active": False # For basic deduplication
            }
            await asyncio.to_thread(self._save_state_sync)
            logger.debug(f"Task initialized with paths: app='{app_file_path}', test='{test_file_path}'")

        await self.task_queue.put(abs_file_path)
        logger.info(f"Design card {abs_file_path} added to queue.")


app = Flask(__name__)
orchestration_service_instance = OrchestrationService()


@app.route("/notify/file_change", methods=["POST"])
async def file_change_notification():
    """Endpoint for File Watcher Service to notify about file changes."""
    data = request.json
    event_type = data.get("event_type")
    file_path = data.get("file_path")

    if not file_path or not event_type:
        return jsonify({"status": "error", "message": "Missing file_path or event_type"}), 400

    logger.info(f"Received file change notification: {event_type} - {file_path}")

    abs_file_path = orchestration_service_instance._get_abs_path(file_path)

    # Use configurable DESIGN_CARDS_DIR for checking if it's a design card
    is_design_card = abs_file_path.startswith(orchestration_service_instance._get_abs_path(orchestration_service_instance.DESIGN_CARDS_DIR + os.sep)) and \
                     (abs_file_path.endswith(".md") or abs_file_path.endswith(".json"))

    if not is_design_card:
        logger.debug(f"Ignoring file change for non-design card path: {abs_file_path}")
        return jsonify(
            {"status": "ignored", "message": "Not a monitored design card path"}
        ), 200


    if event_type == "created":
        with orchestration_service_instance.lock:
            if abs_file_path not in orchestration_service_instance.active_tasks:
                try:
                    content = await orchestration_service_instance._read_file_content(abs_file_path)
                    await orchestration_service_instance.add_design_card(abs_file_path, content)
                    return jsonify({"status": "success", "message": f"Design card {abs_file_path} added to queue."}), 200
                except Exception as e:
                    logger.error(f"Failed to add design card {abs_file_path}: {e}", exc_info=True)
                    return jsonify({"status": "error", "message": f"Failed to add design card: {e}"}), 500
            else:
                logger.info(f"Design card {abs_file_path} already active. Re-queuing if in final state.")
                task_context = orchestration_service_instance.active_tasks[abs_file_path]
                if task_context["status"] in ["TASK_COMPLETE", "TASK_FAILED"]:
                    logger.info(f"Re-queuing previously {task_context['status']} design card {abs_file_path} due to creation event.")
                    task_context["status"] = "PENDING"
                    task_context["iteration_count"] = 0
                    task_context["last_test_results"] = {"status": "NOT_STARTED", "output": ""}
                    task_context["last_code_review_feedback"] = {"status": "NOT_STARTED", "summary": ""}
                    task_context["processing_lock_active"] = False # Reset lock for re-processing
                    await asyncio.to_thread(orchestration_service_instance._save_state_sync)
                    await orchestration_service_instance.task_queue.put(abs_file_path)
                return jsonify({"status": "info", "message": f"Design card {abs_file_path} already active or re-queued."}), 200

    elif event_type == "modified":
        with orchestration_service_instance.lock:
            if abs_file_path in orchestration_service_instance.active_tasks:
                task_context = orchestration_service_instance.active_tasks[abs_file_path]
                # Added: Check processing_lock_active to prevent immediate re-queueing of currently processing tasks
                if task_context.get("processing_lock_active", False):
                    logger.info(f"Modified design card {abs_file_path} is currently being processed. Ignoring this modification event for immediate re-queueing.")
                    return jsonify({"status": "info", "message": f"Design card {abs_file_path} currently being processed; modification deferred."}), 200

                if task_context["status"] in ["TASK_COMPLETE", "TASK_FAILED"]:
                    logger.info(f"Modified completed/failed design card {abs_file_path}. Re-queueing as PENDING.")
                    task_context["status"] = "PENDING"
                    task_context["iteration_count"] = 0
                    task_context["last_test_results"] = {"status": "NOT_STARTED", "output": ""}
                    task_context["last_code_review_feedback"] = {"status": "NOT_STARTED", "summary": ""}
                    content = await orchestration_service_instance._read_file_content(abs_file_path)
                    task_context["design_card_content"] = content
                    task_context["processing_lock_active"] = False # Reset lock
                    await asyncio.to_thread(orchestration_service_instance._save_state_sync)
                    await orchestration_service_instance.task_queue.put(abs_file_path)
                elif task_context["status"] == "IN_PROGRESS":
                    logger.info(f"Modified active design card {abs_file_path}. Re-queuing to ensure re-evaluation.")
                    content = await orchestration_service_instance._read_file_content(abs_file_path)
                    task_context["design_card_content"] = content
                    task_context["processing_lock_active"] = False # Reset lock
                    await asyncio.to_thread(orchestration_service_instance._save_state_sync)
                    await orchestration_service_instance.task_queue.put(abs_file_path)
                else:
                     logger.info(f"Modified design card {abs_file_path} is PENDING. Updating content and ensuring re-evaluation.")
                     content = await orchestration_service_instance._read_file_content(abs_file_path)
                     task_context["design_card_content"] = content
                     task_context["processing_lock_active"] = False # Reset lock
                     await asyncio.to_thread(orchestration_service_instance._save_state_sync)
                     await orchestration_service_instance.task_queue.put(abs_file_path)

                return jsonify({"status": "success", "message": f"Design card {abs_file_path} re-queued for processing."}), 200
            else:
                logger.info(f"Modified file {abs_file_path} is not an active design card. Checking if it's a new design card.")
                try:
                    content = await orchestration_service_instance._read_file_content(abs_file_path)
                    await orchestration_service_instance.add_design_card(abs_file_path, content)
                    return jsonify({"status": "success", "message": f"Design card {abs_file_path} added to queue (via modified event)."}, 200)
                except Exception as e:
                    logger.error(f"Failed to add design card {abs_file_path} via modified event: {e}", exc_info=True)
                    return jsonify({"status": "error", "message": f"Failed to add design card: {e}"}), 500

    elif event_type == "deleted":
        with orchestration_service_instance.lock:
            if abs_file_path in orchestration_service_instance.active_tasks:
                logger.info(f"Design card {abs_file_path} deleted. Removing from active tasks state.")
                del orchestration_service_instance.active_tasks[abs_file_path]
                await asyncio.to_thread(orchestration_service_instance._save_state_sync)
                return jsonify({"status": "success", "message": f"Design card {abs_file_path} removed."}), 200
            else:
                return jsonify({"status": "info", "message": f"Design card {abs_file_path} not found in active tasks."}), 200
    else:
        return jsonify({"status": "info", "message": f"Unhandled event type: {event_type}"}), 200

    return jsonify({"status": "success", "message": "Notification processed."}), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":

    async def main_async_orchestrator_worker():
        # Store the main asyncio event loop instance
        orchestration_service_instance.main_asyncio_loop = asyncio.get_running_loop()
        logger.info("Orchestration Service: Main asyncio loop captured.")

        # Load state and re-queue any in-progress tasks
        await orchestration_service_instance._initial_load_and_queue_tasks()

        # Start the main task processing loop
        await orchestration_service_instance.process_queued_tasks()

    # Helper function to run the async worker in a new thread
    def start_async_loop_in_thread():
        asyncio.run(main_async_orchestrator_worker())

    queue_thread = threading.Thread(target=start_async_loop_in_thread)
    queue_thread.daemon = True
    queue_thread.start()
    logger.info(
        "Orchestration Service: Async task queue processor started in a separate thread."
    )

    PORT = int(os.getenv("ORCHESTRATION_SERVICE_PORT", 5006))
    orchestration_service_instance.LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_communication_service:5001")
    orchestration_service_instance.FILE_SYSTEM_SERVICE_URL = os.getenv("FILE_SYSTEM_SERVICE_URL", "http://file_system_service:5002")
    orchestration_service_instance.TEST_EXECUTION_SERVICE_URL = os.getenv("TEST_EXECUTION_SERVICE_URL", "http://test_execution_service:5004")

    logger.info(f"Orchestration Service starting Uvicorn server on port {PORT}.")
    asgi_app = WSGIMiddleware(app)

    uvicorn.run(
        asgi_app,
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv("LOG_LEVEL_UVICORN", "info").lower(),
    )