# AI_coder_meta_agent/llm_task_definitions.py

# You might have a design_card_schema.py if you're using Pydantic for input parsing
# from design_card_schema import DesignCardMetadata

LLM_TASK_DEFINITIONS = {
    "PLANNING": {
        "role": "Highly Intelligent AI System Architect and Development Strategist",
        "description": (
            "Your critical task is to analyze the provided context (design card, current code state, test results, and "
            "code review feedback) to determine the most effective *next development action*. "
            "Instead of merely selecting a template key, you MUST generate the *complete and precise prompt* "
            "for the selected next LLM task (Coder, Tester, Reviewer, or Debugger). "
            "Ensure the generated prompt is clear, unambiguous, and includes all necessary context from input_data. "
            "Your output will be a JSON object specifying the next task type, the generated prompt, and your reasoning."
        ),
        "model": "gpt-4o-mini",  # Your preferred model for the Planner
        "temperature": 0.0,  # Keep low for deterministic decision-making and prompt generation
        "input_keys": [
            "design_card_content",  # The input design card (MD or JSON)
            "current_app_code",  # Current state of the application code
            "current_test_code",  # Current state of the test code
            "test_results",  # Results from the last test run
            "code_review_feedback",  # Feedback from the last code review
            "iteration_count",  # Current iteration number for the task
            "previous_llm_responses",  # History of Planner decisions and outcomes
            "available_templates",  # For Planner's awareness of available downstream tasks
            "current_system_architecture_overview",  # NEW input for Planner to understand overall system context (e.g., folder structure, other services)
        ],
        "output_guidance": (
            "Your response MUST be a single JSON object. This JSON object MUST be enclosed within a markdown code block, like this: ```json\n{...}\n```\n"  # <-- STRONGER EMPHASIS
            "The JSON object MUST have the following keys:\n"
            "{\n"  # Removed the opening brace here as it's part of the markdown block guidance
            '  "next_llm_task_type": "<one of: TEST_GENERATION, CODE_GENERATION, CODE_REVIEW, DEBUG_CODE_GENERATION, REFACTOR_CODE, GENERATE_SUB_DESIGN_CARD, REFACTOR_SUB_DESIGN_CARD, TASK_COMPLETE, TASK_FAILED>",\n'  # Added sub-design task types
            '  "generated_prompt": "<The COMPLETE prompt string for the next LLM task, including all context and explicit instructions>",\n'
            '  "generated_sub_design_card_json": { /* optional: A complete JSON object representing the new or refactored sub-design card */ },\n'  # For GENERATE/REFACTOR_SUB_DESIGN_CARD
            "  \"debug_target_file\": \"<optional: 'app_file' or 'test_file' if next_llm_task_type is DEBUG_CODE_GENERATION, indicates which code to modify>\",\n"
            '  "reasoning": "<brief explanation of your decision and how the prompt/sub-design card was constructed>"\n'
            "}\n"  # Removed the closing brace here
            "Remember, the entire JSON object MUST be wrapped in ```json\\n...\\n```."  # <-- REPEATED EMPHASIS
            "The `generated_prompt` MUST be a valid and complete prompt for the chosen `next_llm_task_type`. "
            "If `next_llm_task_type` is GENERATE_SUB_DESIGN_CARD or REFACTOR_SUB_DESIGN_CARD, `generated_sub_design_card_json` MUST be present and valid. "
            "For tasks like TASK_COMPLETE or TASK_FAILED, the `generated_prompt`, `generated_sub_design_card_json`, and `debug_target_file` keys should be omitted."
            "Ensure all necessary context (design card, current code, test results, review feedback) is explicitly embedded within the `generated_prompt` for downstream LLMs."
        ),
        "critical_constraints": [
            "- Your ONLY output is a single JSON object. This JSON object MUST be enclosed in a markdown code block (```json\\n...\\n```). NO OTHER TEXT. ALWAYS VALID JSON.",
            "- If `next_llm_task_type` is 'TASK_COMPLETE' or 'TASK_FAILED', omit 'generated_prompt' and 'debug_target_file' and 'generated_sub_design_card_json'.",
            "- If `iteration_count` >= 10 (or MAX_TASK_ITERATIONS from env var), set `next_llm_task_type` to 'TASK_FAILED'.",
            "- The `generated_prompt` must be well-formed and contain all relevant context for the downstream LLM.",
            "- When generating a prompt for CODE_GENERATION, TEST_GENERATION, or DEBUG_CODE_GENERATION, ALWAYS include 'Current Application Code:', 'Current Test Code:', 'Test Results:', and 'Code Review Feedback:' sections using markdown code blocks, embedding the actual content from the input_data.",
            "- For any task requiring code modification (CODE_GENERATION, DEBUG_CODE_GENERATION), the generated prompt MUST conclude with clear output guidance for the LLM to only provide the modified Python code within a markdown block (e.g., '```python\\n[YOUR CODE HERE]\\n```'). For TEST_GENERATION, the output should be the test code in a markdown block.",
            # --- HIGHEST PRIORITY: Missing App File ---
            # If tests fail because the app file is missing, create the app file first.
            "- IF `test_results.status` is 'Fail' AND 'No such file or directory' in `test_results.output` AND `current_app_code` is empty, THEN:",
            "-   - Set `next_llm_task_type` to `CODE_GENERATION`.",
            "-   - Set `debug_target_file` to `'app_file'`.",
            "-   - The `generated_prompt` MUST instruct the Code Generation LLM to implement the required functionality in `current_app_code` based on the `design_card_content` to make the tests pass.",
            "-   - The `reasoning` should clearly state that the application file is missing and needs to be created to resolve the test failures.",
            # --- High-Priority Test Fix: Addressing the "AttributeError: sys.exit_code" loop ---
            # This comes after missing app file, as the app file being there is a prerequisite for these tests to even run meaningfully.
            "- ELSE IF `test_results.status` is 'Fail' AND 'AttributeError: module \\'sys\\' has no attribute \\'exit_code\\'' in `test_results.output`, THEN:",
            "-   - Set `next_llm_task_type` to `DEBUG_CODE_GENERATION`.",
            "-   - Set `debug_target_file` to `'test_file'`.",
            "-   - The `generated_prompt` MUST explicitly instruct the Debugging LLM to refactor the test file (`current_test_code`) to replace `sys.exit_code` assertions with `subprocess.run` and `process.returncode` (or `pytest.raises(SystemExit)` and `excinfo.value.code`).",
            "-   - The `reasoning` MUST highlight that the test code itself needs correction for exit code assertions.",
            # --- High-Priority Code Review Fix: Addressing the "Unredacted Sensitive Data" flaw ---
            "- ELSE IF `code_review_feedback.status` is 'FAILED' AND 'Unredacted Sensitive Data in Exception Logging' in `code_review_feedback.summary`, THEN:",
            "-   - Set `next_llm_task_type` to `DEBUG_CODE_GENERATION`.",
            "-   - Set `debug_target_file` to `'app_file'`.",
            '-   - The `generated_prompt` MUST explicitly instruct the Debugging LLM to modify `current_app_code` by changing `logger.debug(f"[DEBUG-TEMP] Failed to convert arguments to integers. Exception: {e}")` to `logger.debug("[DEBUG-TEMP] Failed to convert arguments to integers.")`. ',
            "-   - The `reasoning` MUST state that the unredacted logging in the app code needs immediate correction.",
            # --- Specific Architectural Task Handling (for high-level decomposition) ---
            "- ELSE IF `design_card_content` (when parsed as JSON) has a 'name' field equal to 'Refactor Data Manager to Service-Orientured Architecture (SOA)' AND no 'sub_design_cards' have been generated in previous_llm_responses for this task, THEN:",
            "-   - Set `next_llm_task_type` to 'GENERATE_SUB_DESIGN_CARD'.",
            "-   - The `generated_sub_design_card_json` MUST contain an array of `sub_design_cards` following the structure discussed (e.g., svc_polygon_fetcher, svc_local_csv_fetcher, svc_data_api_gateway, update_trading_bot, update_docker_compose, cleanup_old_data_manager), including detailed `api_specifications` and `dependencies`.",
            "-   - The `reasoning` must explain the decomposition strategy.",
            "-   - The Planner MUST analyze the `current_system_architecture_overview` to understand the existing code base (`data_manager/data_api.py`, `data_manager/data_fetcher.py`) and use it to inform the implementation details for the new services.",
            # --- General TDD Flow for Initial States or Functional Failures (these apply if above specific rules don't match) ---
            # Initial State: Generate tests first for a new feature/service
            "- ELSE IF `current_app_code` is empty AND `current_test_code` is empty (initial development state for a service/feature), THEN:",
            "-   - Set `next_llm_task_type` to `TEST_GENERATION`.",
            "-   - The `generated_prompt` should instruct the Test Generation LLM to create comprehensive tests based on the `design_card_content`.",
            "-   - The `reasoning` should reflect initiating the TDD RED phase for a new component.",
            # --- NEW RULE ADDED HERE: Trigger Code Review if tests pass AND review hasn't started ---
            "- ELSE IF `test_results.status` is 'Pass' AND `code_review_feedback.status` == 'NOT_STARTED', THEN:",
            "-   - Set `next_llm_task_type` to `CODE_REVIEW`.",
            "-   - The `generated_prompt` should instruct the Reviewer LLM to perform an initial, general code review of `current_app_code` and `current_test_code` based on `design_card_content` and `test_results`.",
            "-   - The `reasoning` should indicate initiating the first code review after tests have passed.",
            # Test Failure (functional, not covered by specific rules above): Generate application code to fix
            # This general functional failure rule now comes *after* the specific "no such file" rule.
            "- ELSE IF `test_results.status` is 'Fail' (and not covered by the specific `AttributeError` rule above, implying a functional bug in app code), THEN:",
            "-   - Set `next_llm_task_type` to `CODE_GENERATION`.",
            "-   - Set `debug_target_file` to `'app_file'`.",  # Default target for app code modifications
            "-   - The `generated_prompt` should instruct the Code Generation LLM to implement/modify `current_app_code` to satisfy requirements and pass `test_results`.",
            "-   - The `reasoning` should indicate addressing functional test failures.",
            # Code Review Failure (general, not covered by specific rules above): Refine application code
            "- ELSE IF `code_review_feedback.status` is 'FAILED' (and not covered by the specific 'Unredacted Sensitive Data' rule above, implying general code quality/style issues), THEN:",
            "-   - Set `next_llm_task_type` to `CODE_GENERATION`.",
            "-   - Set `debug_target_file` to `'app_file'`.",  # Default target
            "-   - The `generated_prompt` should instruct the Code Generation LLM to refine `current_app_code` based on `code_review_feedback`.",
            "-   - The `reasoning` should reflect addressing code review feedback.",
            # All clear (tests pass, review passes): Task Complete or Code Review
            "- ELSE IF `test_results.status` is 'Pass' AND `code_review_feedback.status` is 'PASSED', THEN:",
            "-   - Set `next_llm_task_type` to 'TASK_COMPLETE'.",
            "-   - The `reasoning` should state that all criteria are met for this specific design card.",
            # Fallback: If tests pass but no review done, or general unclear state, initiate a code review
            "- ELSE:",
            "-   - Set `next_llm_task_type` to `CODE_REVIEW`.",
            "-   - The `generated_prompt` should instruct the Reviewer LLM to perform a general code review of `current_app_code` and `current_test_code` based on `design_card_content` and `test_results`.",
            "-   - The `reasoning` should indicate a general review/verification step.",
        ],
    },
    # --- Simplified Task Definitions (their prompts are now generated by PLANNING) ---
    # These tasks will receive their full prompt from the Planner and input_data is typically empty
    "TEST_GENERATION": {
        "role": "Master Test Engineer | TDD & Comprehensive Test Strategy",
        "description": "This task's prompt is dynamically generated by the Planner. It generates or modifies test code.",
        "model": "gemini-1.5-pro-latest",  # Your preferred model for test generation
        "temperature": 0.5,
        "input_keys": [],
    },
    "CODE_GENERATION": {
        "role": "Senior Python Engineer | Production-Ready Systems",
        "description": "This task's prompt is dynamically generated by the Planner. It generates or modifies application code.",
        "model": "gemini-1.5-pro-latest",  # Your preferred model for code generation
        "temperature": 0.3,
        "input_keys": [],
    },
    "DEBUG_CODE_GENERATION": {
        "role": "Diagnostic Engineer | Root Cause Analysis Specialist",
        "description": "This task's prompt is dynamically generated by the Planner. It modifies code (app or test) to resolve specific issues or add diagnostics.",
        "model": "gemini-1.5-pro-latest",  # Your preferred model for debugging code
        "temperature": 0.1,
        "input_keys": [],
    },
    "CODE_REVIEW": {
        "role": "Expert Code Reviewer | Best Practices & Security",
        "description": "This task's prompt is dynamically generated by the Planner. It performs a general code review.",
        "model": "gpt-4o-mini",  # Your preferred model for general code review
        "temperature": 0.0,
        "input_keys": [],
        # --- NEW: Add output_guidance for CODE_REVIEW ---
        "output_guidance": (
            "Your review MUST be a single JSON object. This JSON object MUST be enclosed within a markdown code block, like this: ```json\n{...}\n```\n"
            "The JSON object MUST have the following keys:\n"
            '  "status": "<one of: PASSED, FAILED>",\n'
            '  "summary": "<A concise summary of the review findings>",\n'
            '  "detailed_feedback": [ { "file": "<filename>", "line": <line_number>, "severity": "<CRITICAL, HIGH, MEDIUM, LOW, INFO>", "description": "<detailed issue description>" } ]\n'
            "}\n"
            "Remember, the entire JSON object MUST be wrapped in ```json\\n...\\n```. NO OTHER TEXT. ALWAYS VALID JSON."
        ),
        # --- END NEW ---

    },
    "CODE_REVIEW_DEBUG_CONTEXT": {
        "role": "Senior Security Engineer | Pragmatic Code Reviewer (Debugging Context)",
        "description": "This task's prompt is dynamically generated by the Planner. It performs a code review focusing on temporary diagnostics and security flaws.",
        "model": "deepseek-reasoner",  # Your preferred model for debug-context code review
        "temperature": 0.0,
        "input_keys": [],
    },
    # ARCHITECTURAL_PLANNING would go here if this planner layer calls an even higher one
    # "ARCHITECTURAL_PLANNING": { ... }
}
