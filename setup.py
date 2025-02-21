"""
Project setup script
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a shell command and log output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return False

def setup_project():
    """Set up the entire project"""
    try:
        # Create virtual environment if it doesn't exist
        if not os.path.exists('venv'):
            logger.info("Creating virtual environment...")
            if not run_command('python -m venv venv'):
                raise Exception("Failed to create virtual environment")

        # Activate virtual environment and install dependencies
        logger.info("Installing dependencies...")
        pip_command = 'pip install -r requirements.txt'
        if sys.platform == 'win32':
            if not run_command(f'venv\\Scripts\\activate && {pip_command}'):
                raise Exception("Failed to install dependencies")
        else:
            if not run_command(f'source venv/bin/activate && {pip_command}'):
                raise Exception("Failed to install dependencies")

        # Set up the database
        logger.info("Setting up database...")
        if not run_command('python scripts/setup_db.py'):
            raise Exception("Failed to set up database")

        # Train the ML model
        logger.info("Training ML model...")
        if not run_command('python app/services/ml_algorithm.py'):
            raise Exception("Failed to train ML model")

        logger.info("""
Project setup completed successfully!

To run the project:
1. Activate the virtual environment:
   - Windows: venv\\Scripts\\activate
   - Unix/MacOS: source venv/bin/activate

2. Start the Flask application:
   python app/main.py

The application will be available at http://localhost:5000
""")

    except Exception as e:
        logger.error(f"Project setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    setup_project() 