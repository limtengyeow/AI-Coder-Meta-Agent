import asyncio  # For asyncio.to_thread for synchronous LLM client calls
import json
import logging
import os
import re  # For extracting markdown code blocks
from logging.handlers import RotatingFileHandler  # Import for logging setup

# Import uvicorn to run the Flask app asynchronously
import uvicorn

# httpx is needed if the LLM clients use it internally for async operations
# Flask imports
from flask import Flask, jsonify, request

# Uvicorn's WSGIMiddleware to serve Flask app (WSGI) with Uvicorn (ASGI)
from uvicorn.middleware.wsgi import WSGIMiddleware

# --- Differentiated Logging Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Root logger captures all levels

# Clear existing handlers to prevent duplicate logs if file is re-run (e.g., during development)
if logger.handlers:
    for handler in logger.handlers[:]:  # Iterate over a copy to safely modify
        logger.removeHandler(handler)

# Console Handler
console_handler = logging.StreamHandler()
console_log_level_str = os.getenv("LOG_LEVEL_CONSOLE", "INFO").upper()
console_handler.setLevel(getattr(logging, console_log_level_str))
console_formatter = logging.Formatter(
    "%(name)s | %(asctime)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File Handler
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "ai_coder_full.log"),
    maxBytes=10 * 1024 * 1024,  # 10 MB per file
    backupCount=5,  # Keep 5 backup files
)
file_log_level_str = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()
file_handler.setLevel(getattr(logging, file_log_level_str))
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s | %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Prevent duplicate logs from external libraries that use their own loggers
logging.getLogger("httpx").setLevel(os.getenv("LOG_LEVEL_HTTPX", "WARNING").upper())
logging.getLogger("urllib3").setLevel(os.getenv("LOG_LEVEL_URLLIB3", "WARNING").upper())
# Add Uvicorn specific loggers to control their output
logging.getLogger("uvicorn").setLevel(os.getenv("LOG_LEVEL_UVICORN", "INFO").upper())
logging.getLogger("uvicorn.access").setLevel(
    os.getenv("LOG_LEVEL_UVICORN_ACCESS", "INFO").upper()
)
logging.getLogger("uvicorn.error").setLevel(
    os.getenv("LOG_LEVEL_UVICORN_ERROR", "WARNING").upper()
)


# Module-specific logger for LLMCommunicationService
llm_comm_logger = logging.getLogger(
    __name__
)  # Renamed to avoid confusion with root logger variable
llm_comm_logger.info("Differentiated logging configured for LLM Communication Service.")
# --- End Differentiated Logging Setup ---


class LLMCommunicationService:
    def __init__(self, orchestrator_url=None):
        self.orchestrator_url = orchestrator_url
        self.api_keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        }
        self.api_base_urls = {
            "gemini": os.getenv(
                "GEMINI_API_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ),
            "openai": os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            "deepseek": os.getenv(
                "DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"
            ),
        }

        # Define tasks expected to return JSON
        self.json_expected_tasks = [
            "PLANNING",
            "CODE_REVIEW",
            "CODE_REVIEW_DEBUG_CONTEXT",
            "GENERATE_SUB_DESIGN_CARD",
            "REFACTOR_SUB_DESIGN_CARD"
        ]

        # --- START MODIFIED IMPORT BLOCK ---
        # This robust import block attempts to handle different ways the module might be run.
        try:
            # Attempt an absolute import first. This assumes 'AI_coder_meta_agent' is
            # the top-level package and is discoverable in Python's sys.path.
            from AI_coder_meta_agent.llm_task_definitions import (
                LLM_TASK_DEFINITIONS as MASTER_TASK_DEFINITIONS,
            )
        except ImportError:
            # If the absolute import fails (e.g., when this file is run directly
            # and not as part of the package structure, or if the package isn't in sys.path),
            # try a direct/relative import from the current directory.
            # This assumes llm_task_definitions.py is in the same directory as llm_communication_service.py
            from llm_task_definitions import (
                LLM_TASK_DEFINITIONS as MASTER_TASK_DEFINITIONS,
            )

        self.master_task_definitions = MASTER_TASK_DEFINITIONS
        # --- END MODIFIED IMPORT BLOCK ---

        # Log configured APIs
        if self.api_keys.get("gemini"):
            llm_comm_logger.info("Gemini API configured.")
        if self.api_keys.get("openai"):
            llm_comm_logger.info("OpenAI API configured.")
        if self.api_keys.get("deepseek"):
            llm_comm_logger.info("DeepSeek API configured.")

    async def _call_llm_api(self, model_name: str, messages: list, temperature: float) -> str:
        """Helper to call the specific LLM API client."""
        from openai import AsyncOpenAI  # Import here to ensure it's available

        try:
            if model_name.startswith("gemini"):
                from google.generativeai import GenerativeModel
                client = GenerativeModel(model_name=model_name)
                gemini_messages = [
                    {
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [msg["content"]],
                    }
                    for msg in messages
                ]
                response = await asyncio.to_thread( # Use asyncio.to_thread for sync Gemini client
                    client.generate_content,
                    gemini_messages,
                    generation_config={"temperature": temperature},
                )
                return response.text
            elif model_name.startswith("gpt"):
                client = AsyncOpenAI(
                    api_key=self.api_keys["openai"],
                    base_url=self.api_base_urls["openai"],
                )
                response = await client.chat.completions.create(
                    model=model_name, messages=messages, temperature=temperature
                )
                return response.choices[0].message.content
            elif model_name.startswith("deepseek"):
                client = AsyncOpenAI(
                    api_key=self.api_keys["deepseek"],
                    base_url=self.api_base_urls["deepseek"],
                )
                response = await client.chat.completions.create(
                    model=model_name, messages=messages, temperature=temperature
                )
                return response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            llm_comm_logger.error(f"Error during LLM API call for model {model_name}: {e}", exc_info=True)
            raise  # Re-raise to be caught by the calling method

    async def _normalize_json_output(self, raw_llm_output: str, task_type: str) -> dict:
        """
        Attempts to normalize raw LLM output into valid JSON using gpt-4o-mini as a healing LLM.
        """
        llm_comm_logger.info(f"Attempting JSON normalization for {task_type} output.")
        normalization_model = "gpt-4o-mini"  # Use gpt-4o-mini as decided
        normalization_role = (
            "Strict JSON Normalizer | Output Healing & Validation Specialist. "
            "Your ONLY task is to correct the provided text into valid JSON according to the given schema. "
            "DO NOT add any commentary, explanations, or extra text. Only provide the valid JSON."
        )
        normalization_temperature = 0.0  # Keep low for deterministic parsing

        # Retrieve the expected JSON schema/guidance for the specific task type
        task_def = self.master_task_definitions.get(task_type)
        # Assuming output_guidance in task_definitions already specifies JSON structure
        expected_json_guidance = task_def.get("output_guidance", "Output a valid JSON object.")
        # Refine guidance to specifically ask for pure JSON, not markdown blocks for the healing LLM
        expected_json_guidance_for_healing = (
            f"The expected JSON structure is:\n```json\n{expected_json_guidance}\n```\n"
            f"Provide ONLY the corrected, valid JSON object. DO NOT wrap it in markdown."
        )

        normalization_prompt = (
            f"The following text was generated by another AI and intended to be a valid JSON object. "
            f"However, it might contain formatting errors or extraneous text. "
            f"Your task is to extract the intended JSON object, correct any formatting errors, and return ONLY the valid JSON. "
            f"Strictly adhere to the following schema guidance:\n\n"
            f"{expected_json_guidance_for_healing}\n\n"
            f"Problematic Text:\n```\n{raw_llm_output}\n```\n\n"
            f"Corrected Valid JSON:"
        )

        messages = [
            {"role": "system", "content": normalization_role},
            {"role": "user", "content": normalization_prompt},
        ]

        try:
            healed_text = await self._call_llm_api(normalization_model, messages, normalization_temperature)
            healed_text = healed_text.strip()  # Ensure no leading/trailing whitespace

            llm_comm_logger.debug(f"[DEBUG_LLM_COMM] Healed text from Normalization LLM:\n{healed_text[:500]}...")

            # Attempt to parse the healed text directly (should be pure JSON)
            parsed_healed_json = json.loads(healed_text)
            llm_comm_logger.info(f"Successfully normalized JSON for {task_type} using healing LLM.")
            return {
                "status": "success",
                "parsed_output": parsed_healed_json,  # Return the parsed dict
                "message": "JSON normalized successfully by healing LLM."
            }

        except json.JSONDecodeError as e:
            llm_comm_logger.error(f"Normalization LLM failed to produce valid JSON for {task_type}: {e}. Healed text: {healed_text[:500]}...", exc_info=True)
            return {
                "status": "error",
                "message": f"Normalization LLM could not produce valid JSON: {e}",
                "error_type": "NORMALIZATION_FAILED"
            }
        except Exception as e:
            llm_comm_logger.error(f"Error during Normalization LLM call for {task_type}: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error during normalization LLM call: {e}",
                "error_type": "NORMALIZATION_FAILED"
            }

    async def request_code(
        self, task_type: str, input_data: dict = None, custom_prompt: str = None
    ) -> dict:
        """
        Requests code generation or review from an LLM based on task type.
        Now accepts a custom_prompt to support dynamic prompting from the Planner.
        Includes robust JSON output normalization.
        """
        llm_comm_logger.debug(
            f"[DEBUG_LLM_COMM] Received request for task: {task_type}. Custom prompt provided: {custom_prompt is not None}"
        )
        if input_data:
            llm_comm_logger.debug(
                f"[DEBUG_LLM_COMM] Input data keys: {list(input_data.keys())}"
            )

        task_def = self.master_task_definitions.get(task_type)
        if not task_def:
            llm_comm_logger.error(
                f"Unknown task type: {task_type}. Check llm_task_definitions.py."
            )
            return {
                "status": "error",
                "generated_code": f"Unknown task type: {task_type}",
            }

        model_name = task_def.get("model")
        role_content = task_def.get("role")
        temperature = task_def.get("temperature", 0.0)

        # --- Dynamic Prompting Logic ---
        if custom_prompt:
            full_prompt = custom_prompt
        else:
            if task_type != "PLANNING":
                llm_comm_logger.error(
                    f"Task '{task_type}' called without custom_prompt. This is an invalid state for dynamic prompting. Input data: {input_data}"
                )
                return {
                    "status": "error",
                    "generated_code": f"Invalid LLM call for task {task_type}: custom_prompt missing.",
                }

            description = task_def.get("description", "")
            output_guidance = task_def.get("output_guidance", "")
            critical_constraints = "\n".join(task_def.get("critical_constraints", []))

            formatted_input_data_parts = []
            if input_data:
                for key, value in input_data.items():
                    if key in ["current_app_code", "current_test_code"]:
                        formatted_input_data_parts.append(
                            f"{key.replace('_', ' ').title()}:\n```python\n{value}\n```"
                        )
                    elif key in [
                        "test_results",
                        "code_review_feedback",
                        "previous_llm_responses",
                        "available_templates",
                    ]:
                        formatted_input_data_parts.append(
                            f"{key.replace('_', ' ').title()}:\n```json\n{json.dumps(value, indent=2)}\n```"
                        )
                    elif key == "design_card_content":
                        formatted_input_data_parts.append(
                            f"{key.replace('_', ' ').title()}:\n{value}"
                        )
                    elif key == "current_system_architecture_overview":
                        formatted_input_data_parts.append(
                            f"{key.replace('_', ' ').title()}:\n{value}"
                        )
                    else:
                        formatted_input_data_parts.append(
                            f"{key.replace('_', ' ').title()}: {value}"
                        )

            input_data_str = "\n".join(formatted_input_data_parts)

            full_prompt = (
                f"{description}\n\n"
                f"--- Input Data ---\n"
                f"{input_data_str}\n\n"
                f"--- Output Guidance ---\n"
                f"{output_guidance}\n\n"
                f"--- Critical Constraints ---\n"
                f"{critical_constraints}"
            )
        # --- End Dynamic Prompting Logic ---

        llm_comm_logger.debug(
            f"[DEBUG_LLM_COMM] Full prompt sent to LLM ({model_name}):\n{full_prompt[:2000]}..."
        )

        messages = [
            {"role": "system", "content": role_content},
            {"role": "user", "content": full_prompt},
        ]

        try:
            # Call the LLM API using the helper method
            generated_text = await self._call_llm_api(model_name, messages, temperature)
            llm_comm_logger.debug(
                f"[DEBUG_LLM_COMM] Raw LLM response from {model_name} (partial):\n{generated_text[:500]}..."
            )

            # --- Output Normalization and JSON Parsing ---
            # 1. Extract content from markdown code blocks
            match = re.search(r"```(?:\w+)?\n(.*?)\n```", generated_text, re.DOTALL)
            extracted_content = (
                match.group(1).strip() if match else generated_text.strip()
            )
            llm_comm_logger.debug(
                f"[DEBUG_LLM_COMM] Extracted content block. Length: {len(extracted_content)} bytes. Preview:\n{extracted_content[:500]}..."
            )

            final_output_content = extracted_content
            normalization_status = "success"
            normalization_message = "Content extracted successfully."
            parsed_json_output = None

            if task_type in self.json_expected_tasks:
                llm_comm_logger.debug(f"[DEBUG_LLM_COMM] Task {task_type} expects JSON. Attempting initial parse.")
                try:
                    parsed_json_output = json.loads(extracted_content)
                    final_output_content = parsed_json_output  # Store the parsed object
                    llm_comm_logger.debug(f"[DEBUG_LLM_COMM] JSON parsed successfully for {task_type} on first attempt.")
                except json.JSONDecodeError:
                    llm_comm_logger.warning(f"[WARNING_LLM_COMM] Initial JSON parse failed for {task_type}. Attempting normalization with healing LLM.")
                    # Call the new normalization method (healing LLM)
                    normalization_result = await self._normalize_json_output(extracted_content, task_type)

                    if normalization_result["status"] == "success":
                        parsed_json_output = normalization_result["parsed_output"]
                        final_output_content = parsed_json_output
                        normalization_message = normalization_result["message"]
                    else:
                        normalization_status = "error"
                        normalization_message = normalization_result["message"]
                        llm_comm_logger.error(f"[ERROR_LLM_COMM] Final normalization failed for {task_type}. Message: {normalization_message}")
                        # If normalization completely fails, return an error from LLM Comm Service
                        return {
                            "status": "error",
                            "generated_code": extracted_content,  # Provide raw content for debugging
                            "error_type": "FINAL_JSON_NORMALIZATION_FAILED",
                            "message": normalization_message
                        }
                except Exception as e:  # Catch any other unexpected errors during initial parse
                    normalization_status = "error"
                    normalization_message = f"Unexpected error during initial JSON parse for {task_type}: {e}. Raw content: {extracted_content[:500]}..."
                    llm_comm_logger.error(f"[ERROR_LLM_COMM] {normalization_message}", exc_info=True)
                    return {
                        "status": "error",
                        "generated_code": extracted_content,
                        "error_type": "INITIAL_JSON_PARSE_ERROR",
                        "message": normalization_message
                    }
            # --- End Output Normalization ---

            response_payload = {
                "status": "success",
                # For JSON-expected outputs, generated_code will now hold the parsed dict
                # For code outputs, generated_code still holds the extracted string
                "generated_code": final_output_content,
                "normalization_status": normalization_status,
                "normalization_message": normalization_message
            }
            llm_comm_logger.debug(
                f"[DEBUG_LLM_COMM] Final response payload for OrchestrationService. Task: {task_type}. Status: {normalization_status}"
            )

            return response_payload

        except Exception as e:
            llm_comm_logger.error(
                f"Error communicating with LLM for task '{task_type}' (model: {model_name}): {e}",
                exc_info=True,
            )
            return {
                "status": "error",
                "generated_code": f"LLM communication error for task '{task_type}': {e}",
                "error_type": "LLM_COMMUNICATION_ERROR"
            }


# Flask app instance
app = Flask(__name__)

# Global instance of the LLMCommunicationService
llm_service_instance = LLMCommunicationService(
    orchestrator_url=None  # Orchestrator URL might not be strictly needed by this service
)


@app.route("/llm/request_code", methods=["POST"])
async def request_code_route():
    """
    HTTP endpoint to receive LLM requests from the Orchestration Service.
    This route must be 'async' since Uvicorn runs an ASGI application.
    """
    data = request.json
    task_type = data.get("task_type")
    input_data = data.get("input_data")
    custom_prompt = data.get("custom_prompt")

    if not task_type:
        return jsonify({"status": "error", "message": "task_type is required"}), 400

    try:
        # Call the instance's async method
        response = await llm_service_instance.request_code(
            task_type, input_data, custom_prompt
        )
        return jsonify(response)
    except Exception as e:
        llm_comm_logger.error(  # Use the module-specific logger
            f"Failed to process LLM request for task {task_type}: {e}", exc_info=True
        )
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    PORT = int(os.getenv("LLM_SERVICE_PORT", 5001))
    llm_comm_logger.info(
        f"LLM Communication Service starting Uvicorn server on port {PORT}."
    )

    # --- CRITICAL FIX: Wrap your Flask app with WSGIMiddleware ---
    asgi_app = WSGIMiddleware(app)

    uvicorn.run(
        asgi_app,  # Now serve the ASGI-wrapped Flask app
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv(
            "LOG_LEVEL_UVICORN", "info"
        ).lower(),  # Control Uvicorn's own log level
    )