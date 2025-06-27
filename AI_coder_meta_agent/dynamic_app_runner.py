import importlib.util
import sys
import logging
import os

# Configure basic logging for this runner script
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(filename)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

# This script expects 2 arguments: app_module_name, app_function_name
# Any subsequent arguments are passed as CLI arguments to the app_function.

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python dynamic_app_runner.py <app_module_name> <app_function_name> [app_args...]")
        sys.exit(1)

    app_module_name = sys.argv[1]  # e.g., 'cli_echo_design_card'
    app_function_name = sys.argv[2]  # e.g., 'main' or 'cli_echo'
    app_args = sys.argv[3:]  # Remaining arguments are for the application function

    logger.info(f"Dynamic App Runner: Attempting to run module '{app_module_name}' function '{app_function_name}' with args: {app_args}")

    try:
        # Get project root from environment or calculate
        project_root = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        logger.info(f"Project root: {project_root}")

        # Find and add valid source directories to sys.path
        source_dirs = ['src', 'app', 'src/app']
        valid_paths = [os.path.join(project_root, d) for d in source_dirs if os.path.isdir(os.path.join(project_root, d))]

        for path in valid_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                logger.info(f"Added source path: {path}")

        # Fallback to project root if no valid paths found
        if not valid_paths and project_root not in sys.path and os.path.isdir(project_root):
            sys.path.insert(0, project_root)
            logger.info(f"Using project root as fallback source: {project_root}")

        # Docker-style module fallback
        docker_style_module = f"app.{app_module_name}"
        spec = importlib.util.find_spec(app_module_name) or importlib.util.find_spec(docker_style_module)

        if not spec:
            logger.error(f"Module '{app_module_name}' not found (tried: '{app_module_name}', '{docker_style_module}')")
            logger.error("Searched paths: " + ", ".join(sys.path))
            sys.exit(1)

        # Use Docker-style name if found
        if spec.name.startswith("app."):
            app_module_name = spec.name
            logger.info(f"Using Docker-style module: {app_module_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function/entry point from the module
        if not hasattr(module, app_function_name):
            logger.error(f"Function '{app_function_name}' not found in module '{app_module_name}'")
            sys.exit(1)

        target_function = getattr(module, app_function_name)

        # Prepare execution environment
        original_sys_argv = sys.argv
        sys.argv = [app_module_name] + app_args  # More meaningful script name

        try:
            logger.info(f"Executing '{app_function_name}' with arguments: {app_args}")
            result = target_function()

            # Handle exit code if function returns an integer
            if isinstance(result, int):
                logger.info(f"Function returned exit code: {result}")
                sys.exit(result)
            else:
                logger.info("Function executed successfully")
                sys.exit(0)

        finally:
            sys.argv = original_sys_argv  # Restore original sys.argv

    except ImportError as e:
        logger.error(f"ImportError: {str(e)}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)