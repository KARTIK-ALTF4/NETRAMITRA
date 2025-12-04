"""
Database migration script to add new profile fields to User table
"""
import sqlite3
import os

db_path = 'instance/cataract_detection.db'

if not os.path.exists(db_path):
    print("❌ Database file not found!")
    print(f"Looking for: {os.path.abspath(db_path)}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check current schema
cursor.execute("PRAGMA table_info(user)")
columns = [col[1] for col in cursor.fetchall()]
print(f"Current columns: {columns}")

# Add new columns if they don't exist
new_columns = [
    ('age_group', 'VARCHAR(20)'),
    ('gender', 'VARCHAR(20)'),
    ('occupation', 'VARCHAR(100)'),
    ('medical_conditions', 'TEXT'),
    ('eye_problems_history', 'TEXT'),
    ('family_history', 'TEXT')
]

for col_name, col_type in new_columns:
    if col_name not in columns:
        try:
            cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} {col_type}")
            print(f"✅ Added column: {col_name}")
        except sqlite3.OperationalError as e:
            print(f"⚠️ Column {col_name} already exists or error: {e}")

conn.commit()
conn.close()

print("\n✅ Database migration completed successfully!")
print("You can now restart the server.")
