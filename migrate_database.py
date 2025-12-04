"""
Database migration script to add phone and gradcam_path columns
"""
import sqlite3
import os

db_path = 'instance/cataract_detection.db'

if not os.path.exists(db_path):
    print("❌ Database file not found! Run app.py first to create database.")
    exit(1)

print("🔧 Migrating database schema...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if phone column exists in user table
    cursor.execute("PRAGMA table_info(user)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'phone' not in columns:
        print("➕ Adding 'phone' column to user table...")
        cursor.execute("ALTER TABLE user ADD COLUMN phone VARCHAR(15)")
        print("✅ Added 'phone' column")
    else:
        print("ℹ️ 'phone' column already exists")
    
    # Check if gradcam_path column exists in prediction table
    cursor.execute("PRAGMA table_info(prediction)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'gradcam_path' not in columns:
        print("➕ Adding 'gradcam_path' column to prediction table...")
        cursor.execute("ALTER TABLE prediction ADD COLUMN gradcam_path VARCHAR(200)")
        print("✅ Added 'gradcam_path' column")
    else:
        print("ℹ️ 'gradcam_path' column already exists")
    
    conn.commit()
    print("\n✅ Database migration completed successfully!")
    
except Exception as e:
    print(f"❌ Error during migration: {e}")
    conn.rollback()
finally:
    conn.close()

print("\n📊 Verifying database schema...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("\n👥 User table columns:")
cursor.execute("PRAGMA table_info(user)")
for row in cursor.fetchall():
    print(f"   - {row[1]} ({row[2]})")

print("\n🔍 Prediction table columns:")
cursor.execute("PRAGMA table_info(prediction)")
for row in cursor.fetchall():
    print(f"   - {row[1]} ({row[2]})")

conn.close()
print("\n✅ Migration complete! You can now restart the app.")
