from flask import Flask, request, jsonify
import subprocess
import os
import logging
import sys

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Docker-compatible configuration
PROJECT_ROOT = os.getenv('PROJECT_ROOT', '/project')
logger.info(f"Test Execution Service: Project root set to: {PROJECT_ROOT}")

@app.route('/tests/run', methods=['POST'])
def execute_tests():
    """Execute pytest tests and return results"""
    data = request.get_json()
    test_context = data.get('test_context')

    if not test_context:
        return jsonify({"error": "No test_context provided"}), 400

    abs_test_path = os.path.join(PROJECT_ROOT, test_context)
    if not os.path.abspath(abs_test_path).startswith(os.path.abspath(PROJECT_ROOT)):
        logger.warning(f"Security Alert: Attempted to run test outside PROJECT_ROOT: {test_context}")
        return jsonify({"error": "Invalid test path: Must be within project root"}), 403

    logger.info(f"Running pytest for: {test_context}")

    # Prepare pytest command
    pytest_command = [
        sys.executable,  # Use current Python interpreter
        "-m", "pytest",
        abs_test_path
    ]

    test_stdout = ""
    test_stderr = ""
    overall_result = "Error"
    application_logs = ""

    try:
        # Execute pytest
        pytest_process = subprocess.run(
            pytest_command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        test_stdout = pytest_process.stdout
        test_stderr = pytest_process.stderr

        if pytest_process.returncode == 0:
            overall_result = "Pass"
            logger.info("Pytest completed successfully")
        else:
            overall_result = "Fail"
            logger.warning(f"Pytest failed with exit code: {pytest_process.returncode}")

    except subprocess.TimeoutExpired:
        logger.error("Pytest execution timed out after 5 minutes")
        test_stderr = "Test execution timed out"
    except Exception as e:
        logger.error(f"Pytest execution failed: {str(e)}", exc_info=True)
        test_stderr = f"Execution error: {str(e)}"

    return jsonify({
        "status": "success",
        "overall_result": overall_result,
        "stdout": test_stdout,
        "stderr": test_stderr,
        "application_logs": application_logs
    }), 200

if __name__ == '__main__':
    PORT = int(os.getenv("TEST_EXECUTION_SERVICE_PORT", 5004))
    logger.info(f"Starting Test Execution Service on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)