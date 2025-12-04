"""
Check database status - users and predictions
"""
from app import app, db, User, Prediction

with app.app_context():
    print("=" * 60)
    print("DATABASE STATUS CHECK")
    print("=" * 60)
    
    # Check users
    users = User.query.all()
    print(f"\n👥 Total Users: {len(users)}")
    for user in users:
        print(f"   - {user.username} ({user.email})")
        if user.phone:
            print(f"     Phone: {user.phone}")
    
    # Check predictions
    predictions = Prediction.query.all()
    print(f"\n🔍 Total Predictions: {len(predictions)}")
    for pred in predictions:
        user = User.query.get(pred.user_id)
        print(f"   - ID: {pred.id}")
        print(f"     User: {user.username if user else 'Unknown'}")
        print(f"     Result: {pred.prediction_result}")
        print(f"     Confidence: {pred.confidence*100:.2f}%")
        print(f"     Timestamp: {pred.timestamp}")
        print(f"     Image: {pred.image_path}")
        print(f"     Grad-CAM: {pred.gradcam_path if pred.gradcam_path else 'Not available'}")
        print()
    
    if len(users) == 0:
        print("\n⚠️ No users found! Please register first.")
    
    if len(predictions) == 0:
        print("\n⚠️ No predictions found! Please upload an image after logging in.")
    
    print("=" * 60)
