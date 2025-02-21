"""
Pytest configuration and fixtures.
"""

import pytest
from app import create_app
from app.models import Book
from app.services import BookRecommendationChatbot, BookVectorStore

@pytest.fixture
def app():
    """Create and configure a test Flask application."""
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

@pytest.fixture
def chatbot():
    """Create a test chatbot instance."""
    return BookRecommendationChatbot(db_path='data/books.db')

@pytest.fixture
def vector_store():
    """Create a test vector store instance."""
    return BookVectorStore(db_path='data/books.db') 