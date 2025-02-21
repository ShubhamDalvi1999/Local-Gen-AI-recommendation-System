"""
Database initialization script.
This script creates the database tables and loads initial data from CSV files.
"""

import os
import sys
import csv
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text
from app.models.book import Base, Book
from app.core.config import DATABASE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database and load initial data."""
    # Create database engine
    engine = create_engine(DATABASE['default']['URL'], echo=DATABASE['default']['ECHO'])
    
    try:
        # Drop existing tables
        Base.metadata.drop_all(engine)
        
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Load initial data
        data_file = Path(__file__).resolve().parent.parent / 'data' / 'books' / 'data.csv'
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            return
        
        # Read and insert data
        with engine.connect() as conn:
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Convert empty strings to None
                        for key, value in row.items():
                            if value == '':
                                row[key] = None
                            elif key in ['published_year', 'num_pages', 'ratings_count']:
                                row[key] = int(value) if value else None
                            elif key == 'average_rating':
                                row[key] = float(value) if value else None
                        
                        # Insert data
                        stmt = Book.__table__.insert().values(
                            title=row.get('title'),
                            link=row.get('link'),
                            description=row.get('description'),
                            isbn13=row.get('isbn13'),
                            isbn10=row.get('isbn10'),
                            subtitle=row.get('subtitle'),
                            authors=row.get('authors'),
                            categories=row.get('categories'),
                            thumbnail=row.get('thumbnail'),
                            published_year=row.get('published_year'),
                            average_rating=row.get('average_rating'),
                            num_pages=row.get('num_pages'),
                            ratings_count=row.get('ratings_count')
                        )
                        conn.execute(stmt)
                        
                    except Exception as e:
                        logger.error(f"Error inserting row: {row.get('title', 'Unknown')}. Error: {str(e)}")
                        continue
                
                conn.commit()
                logger.info("Initial data loaded successfully")
                
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

if __name__ == '__main__':
    init_db() 