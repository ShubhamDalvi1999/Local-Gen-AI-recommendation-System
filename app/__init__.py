"""
Flask application package.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.core.config import DATABASE

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    
    # Configure the Flask application
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE['default']['URL']
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    from app.api.routes.chat_routes import chat_bp
    from app.api.routes.main_routes import main_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    
    return app 