from flask import Flask, request, jsonify
import os
import logging

app = Flask(__name__)

# Configure logging for the service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CHANGED: Use PROJECT_ROOT from environment variable (Docker compatible)
PROJECT_ROOT = os.getenv('PROJECT_ROOT', '/project')  # Default to /project in containers
logger.info(f"File System Service: Project root set to: {PROJECT_ROOT}")

@app.route('/files/write', methods=['POST'])
def write_file():
    """Write/update a file relative to PROJECT_ROOT"""
    data = request.get_json()
    file_path = data.get('file_path')
    content = data.get('content')

    if not file_path or content is None:
        return jsonify({"error": "Missing file_path or content"}), 400

    # Security: Prevent path traversal
    abs_file_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.abspath(abs_file_path).startswith(os.path.abspath(PROJECT_ROOT)):
        logger.warning(f"Security Alert: Attempted write outside PROJECT_ROOT: {file_path}")
        return jsonify({"error": "Invalid file path: Must be within project root"}), 403

    try:
        os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        with open(abs_file_path, 'w') as f:
            f.write(content)
        logger.info(f"Successfully wrote to {file_path}")
        return jsonify({"status": "success", "message": f"File {file_path} written"}), 200
    except IOError as e:
        logger.error(f"Write failed: {e}", exc_info=True)
        return jsonify({"error": f"Failed to write file: {str(e)}"}), 500

@app.route('/files/read', methods=['POST'])
def read_file():
    """Read a file relative to PROJECT_ROOT"""
    data = request.get_json()
    file_path = data.get('file_path')

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    abs_file_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.abspath(abs_file_path).startswith(os.path.abspath(PROJECT_ROOT)):
        logger.warning(f"Security Alert: Attempted read outside PROJECT_ROOT: {file_path}")
        return jsonify({"error": "Invalid file path: Must be within project root"}), 403

    try:
        with open(abs_file_path, 'r') as f:
            content = f.read()
        logger.info(f"Successfully read {file_path}")
        return jsonify({"status": "success", "content": content}), 200
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {file_path}"}), 404
    except IOError as e:
        logger.error(f"Read failed: {e}", exc_info=True)
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500

@app.route('/files/exists', methods=['POST'])
def file_exists():
    """Check if a file exists relative to PROJECT_ROOT"""
    data = request.get_json()
    file_path = data.get('file_path')

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    abs_file_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.abspath(abs_file_path).startswith(os.path.abspath(PROJECT_ROOT)):
        logger.warning(f"Security Alert: Attempted existence check outside PROJECT_ROOT: {file_path}")
        return jsonify({"error": "Invalid file path: Must be within project root"}), 403

    exists = os.path.exists(abs_file_path)
    logger.info(f"Existence check for {file_path}: {exists}")
    return jsonify({"status": "success", "exists": exists}), 200

@app.route('/files/delete', methods=['POST'])
def delete_file():
    """Delete a file relative to PROJECT_ROOT"""
    data = request.get_json()
    file_path = data.get('file_path')

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    abs_file_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.abspath(abs_file_path).startswith(os.path.abspath(PROJECT_ROOT)):
        logger.warning(f"Security Alert: Attempted delete outside PROJECT_ROOT: {file_path}")
        return jsonify({"error": "Invalid file path: Must be within project root"}), 403

    try:
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)
            logger.info(f"Deleted {file_path}")
            return jsonify({"status": "success", "message": f"File {file_path} deleted"}), 200
        else:
            logger.info(f"File {file_path} not found")
            return jsonify({"status": "success", "message": "File not found, no action"}), 200
    except OSError as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

@app.route('/files/move', methods=['POST'])
def move_file():
    """Move a file within PROJECT_ROOT"""
    data = request.get_json()
    src_file_path = data.get('src_file_path')
    dest_file_path = data.get('dest_file_path')

    if not src_file_path or not dest_file_path:
        return jsonify({"error": "Missing src_file_path or dest_file_path"}), 400

    abs_src = os.path.join(PROJECT_ROOT, src_file_path)
    abs_dest = os.path.join(PROJECT_ROOT, dest_file_path)

    # Security checks
    if not os.path.abspath(abs_src).startswith(os.path.abspath(PROJECT_ROOT)):
        return jsonify({"error": "Invalid source path"}), 403
    if not os.path.abspath(abs_dest).startswith(os.path.abspath(PROJECT_ROOT)):
        return jsonify({"error": "Invalid destination path"}), 403

    try:
        if not os.path.exists(abs_src):
            logger.warning(f"Source file not found: {src_file_path}")
            return jsonify({"status": "success", "message": "Source not found, no action"}), 200

        os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
        os.rename(abs_src, abs_dest)
        logger.info(f"Moved {src_file_path} to {dest_file_path}")
        return jsonify({"status": "success", "message": f"Moved from {src_file_path} to {dest_file_path}"}), 200
    except OSError as e:
        logger.error(f"Move failed: {e}", exc_info=True)
        return jsonify({"error": f"Failed to move file: {str(e)}"}), 500

if __name__ == '__main__':
    # CHANGED: Use standardized port handling
    PORT = int(os.getenv("FILE_SYS_SERVICE_PORT", 5002))
    logger.info(f"Starting File System Service on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)