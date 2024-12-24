"""
Database portal for querying and managing the SQLite database
"""

from flask import Blueprint, render_template, request, jsonify
import sqlite3
import logging
from typing import List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

db_portal = Blueprint('db_portal', __name__)

def verify_database():
    """Verify database exists and has correct schema"""
    try:
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        
        # Check if Books table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Books'")
        if not cursor.fetchone():
            logger.error("Books table not found in database")
            return False, "Books table not found in database"
            
        # Check table schema
        cursor.execute("PRAGMA table_info(Books)")
        columns = {col[1] for col in cursor.fetchall()}
        required_columns = {'id', 'book', 'link', 'vector', 'description'}
        
        missing_columns = required_columns - columns
        if missing_columns:
            logger.error(f"Missing columns in Books table: {missing_columns}")
            return False, f"Missing columns: {missing_columns}"
            
        conn.close()
        return True, "Database verified"
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False, str(e)

def get_db_connection():
    """Create a database connection"""
    return sqlite3.connect('my_database.db')

def get_table_schema(table_name: str) -> str:
    """Get the schema for a specific table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        schema = cursor.fetchone()
        conn.close()
        return schema[0] if schema else ""
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return ""

def get_tables() -> List[str]:
    """Get list of tables in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return []

def execute_query(query: str) -> Tuple[Optional[List[str]], Optional[List[Tuple[Any, ...]]], Optional[str]]:
    """Execute SQL query and return columns, results, and any error"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Get column names
        columns = [description[0] for description in cursor.description] if cursor.description else []
        
        # Get results and convert BLOB to string representation
        results = []
        for row in cursor.fetchall():
            processed_row = []
            for item in row:
                if isinstance(item, bytes):
                    processed_row.append("<BLOB data>")
                else:
                    processed_row.append(item)
            results.append(tuple(processed_row))
        
        conn.commit()
        conn.close()
        
        return columns, results, None
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None, None, str(e)

@db_portal.route('/db_portal', methods=['GET', 'POST'])
def portal():
    """Database portal interface"""
    logger.info("DB Portal accessed")
    
    # Verify database first
    is_valid, error_msg = verify_database()
    logger.info(f"Database verification: {is_valid}, {error_msg}")
    
    if not is_valid:
        return render_template(
            'db_portal.html',
            error=f"Database Error: {error_msg}",
            tables=[],
            schema="",
            query="",
            columns=None,
            result=None
        )
    
    tables = get_tables()
    logger.info(f"Found tables: {tables}")
    schema = get_table_schema('Books')
    
    query = None
    columns = None
    result = None
    error = False
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            logger.info(f"Executing query: {query}")
            columns, result, error_msg = execute_query(query)
            if error_msg:
                result = error_msg
                error = True
    
    try:
        # Add default query if none provided
        if not query:
            query = "SELECT id, book, link, description, CASE WHEN vector IS NOT NULL THEN '<BLOB data>' ELSE 'NULL' END as vector FROM Books LIMIT 5;"
        
        return render_template(
            'db_portal.html',
            tables=tables,
            schema=schema,
            query=query,
            columns=columns,
            result=result,
            error=error
        )
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        return f"Error rendering template: {e}", 500

@db_portal.route('/db_test')
def test():
    """Test route to verify blueprint is working"""
    return "Database portal is accessible" 