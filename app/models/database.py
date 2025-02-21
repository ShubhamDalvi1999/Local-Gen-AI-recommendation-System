"""
Database configuration and session management
"""

from app import db

def init_db():
    """Initialize the database, creating all tables"""
    db.create_all()

def shutdown_session(exception=None):
    """Remove the session at the end of request"""
    db.session.remove() 