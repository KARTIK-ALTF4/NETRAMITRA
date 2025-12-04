"""
Script to update database schema for gradcam_path field
"""
from app import app, db

with app.app_context():
    print("Updating database schema...")
    
    # Drop all tables and recreate (WARNING: This deletes all data)
    # Uncomment the lines below if you want to reset the database
    # db.drop_all()
    # print("All tables dropped")
    
    # Create all tables with new schema
    db.create_all()
    print("✅ Database schema updated successfully!")
    print("New field 'gradcam_path' added to Prediction table")
