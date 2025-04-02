# Make app directory a package 

from flask import Flask
from flask_cors import CORS
from app.api.routes import api
from app.database.db_manager import init_db

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Initialize database
    init_db()
    
    return app 