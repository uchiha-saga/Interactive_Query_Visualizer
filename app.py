from app.api import create_app
from app.database.db_manager import setup_database

if __name__ == "__main__":
    # Setup database tables and extensions
    try:
        setup_database()
    except Exception as e:
        print(f"Warning: Database setup failed: {e}")
        print("Running with sample data...")
    
    # Create and run the Flask app
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001) 