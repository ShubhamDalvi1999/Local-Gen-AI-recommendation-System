"""
Book model definition using SQLAlchemy ORM.
"""

from app import db

class Book(db.Model):
    """Book model representing a book in the database."""
    
    __tablename__ = 'books'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    link = db.Column(db.String(500))
    description = db.Column(db.Text)
    isbn13 = db.Column(db.String(13))
    isbn10 = db.Column(db.String(10))
    subtitle = db.Column(db.String(500))
    authors = db.Column(db.String(500))
    categories = db.Column(db.String(500))
    thumbnail = db.Column(db.String(500))
    published_year = db.Column(db.Integer)
    average_rating = db.Column(db.Float)
    num_pages = db.Column(db.Integer)
    ratings_count = db.Column(db.Integer)
    embedding = db.Column(db.Text)  # Stored as JSON string

    def __repr__(self):
        """String representation of the book."""
        return f"<Book(id={self.id}, title='{self.title}')>"

    def to_dict(self):
        """Convert book to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'link': self.link,
            'description': self.description,
            'isbn13': self.isbn13,
            'isbn10': self.isbn10,
            'subtitle': self.subtitle,
            'authors': self.authors,
            'categories': self.categories,
            'thumbnail': self.thumbnail,
            'published_year': self.published_year,
            'average_rating': self.average_rating,
            'num_pages': self.num_pages,
            'ratings_count': self.ratings_count
        } 