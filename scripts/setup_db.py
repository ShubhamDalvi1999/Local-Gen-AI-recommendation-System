"""
Database setup and migration script
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import create_app
from app.models.database import init_db

def setup_database():
    """Set up the database and run migrations"""
    try:
        # Create Flask app and push context
        app = create_app()
        
        with app.app_context():
            # Create data directory if it doesn't exist
            data_dir = os.path.join(app.root_path, '..', 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Initialize database
            logger.info("Creating database tables...")
            init_db()
            
            logger.info("Database setup completed successfully!")
            
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        raise

def main():
    """Main entry point"""
    try:
        setup_database()
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 