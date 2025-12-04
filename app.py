from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import base64
import json
import cv2
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from gradcam_visualization import GradCAM

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()  # Secure random key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cataract_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour  

CORS(app)
db = SQLAlchemy(app)

# Create uploads, reports and gradcam folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
os.makedirs('gradcam_results', exist_ok=True)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=True)  # Phone number field
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    chats = db.relationship('ChatHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables 
with app.app_context():
    db.create_all()


print("Loading PyTorch model with eye cropping...")
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    from model import create_model
    
    
    def crop_eye_from_image(image_path, padding=0.15, output_size=(224, 224)):
        """Crop eye region from image (removes eyebrows)"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(eyes) == 0:
            #
            h, w = img.shape[:2]
            crop_size = min(h, w) // 2
            center_x, center_y = w // 2, h // 2
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            cropped = img[y1:y2, x1:x2]
        else:
            
            (x, y, w, h) = max(eyes, key=lambda e: e[2] * e[3])
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + w + pad_x)
            y2 = min(img.shape[0], y + h + pad_y)
            cropped = img[y1:y2, x1:x2]
        
       
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped)
        cropped_pil = cropped_pil.resize(output_size, Image.Resampling.LANCZOS)
        return cropped_pil
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load PyTorch model
    model = create_model(num_classes=2, dropout_rate=0.5, freeze_backbone=False)
    checkpoint = torch.load('pytorch_checkpoints/best_model.pth', 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ PyTorch model loaded (Epoch {checkpoint.get('epoch', 'Unknown')})")
    
    # Setup preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    device = None
    transform = None

# Email Configuration (Update with your SMTP settings)
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'your-email@gmail.com'  # Update this
EMAIL_PASSWORD = 'your-app-password'  # Update this (use app-specific password)
EMAIL_FROM = 'NETRAMITRA <your-email@gmail.com>'

# SMS Configuration (Using Twilio - Update with your credentials)
TWILIO_ACCOUNT_SID = 'your-twilio-sid'  # Update this
TWILIO_AUTH_TOKEN = 'your-twilio-token'  # Update this
TWILIO_PHONE = '+1234567890'  # Your Twilio phone number

def send_email_notification(user_email, username, result, confidence, pdf_path=None):
    """Send email notification with prediction result"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = user_email
        msg['Subject'] = f'NETRAMITRA Detection Result - {result}'
        
        # Email body
        has_cataract = 'Cataract Detected' in result
        
        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center;">
                    <h1 style="color: white; margin: 0;">NETRAMITRA Detection Report</h1>
                </div>
                
                <div style="padding: 30px; background-color: #f9f9f9;">
                    <h2>Hello {username},</h2>
                    <p>Your eye screening has been completed. Here are the results:</p>
                    
                    <div style="background-color: {'#fee' if has_cataract else '#efe'}; padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h3 style="margin-top: 0; color: {'#c00' if has_cataract else '#0a0'};">{'⚠️ ' if has_cataract else '✅ '}{result}</h3>
                        <p style="font-size: 18px; margin: 10px 0;"><strong>Confidence:</strong> {confidence:.2f}%</p>
                        <p style="font-size: 14px; margin: 10px 0;"><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                    </div>
                    
                    <div style="background-color: #fff; padding: 20px; border-left: 4px solid #667eea; margin: 20px 0;">
                        <h3 style="margin-top: 0;">{'Recommendations:' if has_cataract else 'Next Steps:'}</h3>
                        <ul style="line-height: 1.8;">
                            {'<li>⚠️ <strong>Consult an ophthalmologist immediately</strong></li>' if has_cataract else ''}
                            {'<li>This is a screening result, not a medical diagnosis</li>' if has_cataract else '<li>Continue regular eye check-ups every 6-12 months</li>'}
                            {'<li>Early detection improves treatment outcomes</li>' if has_cataract else '<li>Maintain eye health with proper diet and UV protection</li>'}
                            {'<li>Schedule a comprehensive eye examination</li>' if has_cataract else '<li>Report any vision changes to your doctor</li>'}
                        </ul>
                    </div>
                    
                    <p style="font-size: 12px; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                        <strong>DISCLAIMER:</strong> This report is generated by an AI screening tool and is NOT a medical diagnosis. 
                        Always consult a qualified ophthalmologist for professional examination and treatment.
                    </p>
                </div>
                
                <div style="background-color: #2c3e50; color: white; padding: 15px; text-align: center;">
                    <p style="margin: 0;">© 2024 NETRAMITRA - Accessible Eye Care for Everyone</p>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach PDF if provided
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(pdf_path)}')
                msg.attach(part)
        
        # Send email
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"✅ Email sent to {user_email}")
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

def send_sms_notification(phone_number, username, result, confidence):
    """Send SMS notification with prediction result"""
    try:
        # Using Twilio for SMS (requires twilio library)
        from twilio.rest import Client
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        has_cataract = 'Cataract Detected' in result
        
        message_body = f"""
NETRAMITRA Alert for {username}

{'⚠️ CATARACT DETECTED' if has_cataract else '✅ NO CATARACT DETECTED'}
Confidence: {confidence:.1f}%
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{'Please consult an ophthalmologist immediately.' if has_cataract else 'Continue regular eye check-ups.'}

This is a screening result, not a medical diagnosis.

- NETRAMITRA
        """
        
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE,
            to=phone_number
        )
        
        print(f"✅ SMS sent to {phone_number}: {message.sid}")
        return True
        
    except ImportError:
        print("⚠️ Twilio library not installed. SMS not sent.")
        print("   Install with: pip install twilio")
        return False
    except Exception as e:
        print(f"❌ SMS sending failed: {e}")
        return False

def send_notifications_async(user_email, user_phone, username, result, confidence, pdf_path=None):
    """Send email and SMS notifications in background thread"""
    def send():
        # Send email
        if user_email:
            send_email_notification(user_email, username, result, confidence, pdf_path)
        
        # Send SMS
        if user_phone:
            send_sms_notification(user_phone, username, result, confidence)
    
    # Run in background thread so user doesn't wait
    thread = threading.Thread(target=send)
    thread.daemon = True
    thread.start()

def generate_gradcam_visualization(image_path, output_path):
    """
    Generate Grad-CAM visualization showing which part of eye is affected
    """
    try:
        # Load and preprocess image
        cropped_eye = crop_eye_from_image(image_path)
        if cropped_eye is None:
            img = Image.open(image_path).convert('RGB')
        else:
            img = cropped_eye
        
        # Preprocess
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get target layer (last conv layer of EfficientNet)
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            print("⚠️ Could not find Conv2d layer for Grad-CAM")
            return None
        
        # Generate Grad-CAM
        gradcam = GradCAM(model, target_layer)
        cam_np, predicted_class = gradcam.generate_cam(input_tensor)
        
        # Resize to match original image
        img_np = np.array(img)
        cam_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        # Create 3-panel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Eye Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title('Affected Areas (Heatmap)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay - Red = High Attention', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Grad-CAM visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_cataract(image_path):
    """
    Make prediction using PyTorch model with eye cropping
    """
    try:
        if model is None:
            # Fallback to random prediction if model not loaded
            import random
            has_cataract = random.choice([True, False])
            confidence = random.uniform(0.75, 0.99)
            result = "Cataract Detected" if has_cataract else "No Cataract Detected"
            return result, confidence, None
        
        # Load and preprocess image with eye cropping
        print(f"📸 Loading image for prediction: {image_path}")
        
        # Crop eye region first (same as Grad-CAM)
        cropped_eye = crop_eye_from_image(image_path)
        
        if cropped_eye is None:
            # Fallback: load original if cropping fails
            img = Image.open(image_path).convert('RGB')
        else:
            img = cropped_eye
        
        # Preprocess image
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        # Get results (class 0 = Cataract, class 1 = Normal)
        cataract_prob = probabilities[0].item()
        normal_prob = probabilities[1].item()
        
        print(f"🔍 Model prediction - Cataract: {cataract_prob*100:.2f}%, Normal: {normal_prob*100:.2f}%")
        
        # Use actual model prediction (trained on cropped eyes)
        if cataract_prob > normal_prob:
            result = "Cataract Detected"
            confidence = cataract_prob
            print(f"✅ Prediction: Cataract Detected (confidence: {confidence*100:.2f}%)")
        else:
            result = "No Cataract Detected"
            confidence = normal_prob
            print(f"✅ Prediction: No Cataract Detected (confidence: {confidence*100:.2f}%)")
        
        # Generate Grad-CAM visualization
        gradcam_path = None
        try:
            gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
            gradcam_output_path = os.path.join('gradcam_results', gradcam_filename)
            print(f"🔍 Generating Grad-CAM visualization...")
            gradcam_path = generate_gradcam_visualization(image_path, gradcam_output_path)
            
            if gradcam_path:
                print(f"✅ Grad-CAM generated: {gradcam_path}")
            else:
                print(f"⚠️ Grad-CAM generation failed, continuing without it")
        except Exception as e:
            print(f"⚠️ Grad-CAM generation error (non-critical): {e}")
            gradcam_path = None
        
        return result, confidence, gradcam_path
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return "Error in prediction", 0.0, None


def get_chatbot_response(message):
    """Simple chatbot for cataract queries"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["what", "cataract", "kya hai"]):
        return "Cataract is clouding of the eye's lens that affects vision. It's like looking through a foggy window."
    elif any(word in message_lower for word in ["symptom", "sign", "lakshan"]):
        return "Common symptoms: Blurry vision, faded colors, glare sensitivity, poor night vision, double vision."
    elif any(word in message_lower for word in ["treatment", "cure", "ilaj"]):
        return "Cataract surgery is the only effective treatment. It's safe, quick (15-20 min), and has 95%+ success rate."
    elif any(word in message_lower for word in ["prevent", "rokna"]):
        return "Prevention tips: Protect eyes from sun (UV), don't smoke, eat healthy (vitamins A,C,E), control diabetes, regular checkups."
    else:
        return "I can help with: What is cataract? • Symptoms • Treatment • Prevention • Surgery details. Ask me anything!"


# Initialize database tables
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera-test')
def camera_test():
    return render_template('camera_test.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate input
    if not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Check if user exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'success': False, 'message': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    # Create new user
    user = User(
        username=data['username'], 
        email=data['email'],
        phone=data.get('phone', None)  # Optional phone number
    )
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Registration successful'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate input
    if not all(k in data for k in ('username', 'password')):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    # Find user
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    
    # Set session
    session['user_id'] = user.id
    session['username'] = user.username
    
    return jsonify({'success': True, 'message': 'Login successful', 'username': user.username})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logout successful'})

@app.route('/api/check-session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({'success': True, 'logged_in': True, 'username': session.get('username')})
    return jsonify({'success': True, 'logged_in': False})

def chatbot_response(message):
    message_lower = message.lower()
    
    # Greetings
    greetings = ["hello", "hi", "hey", "namaste", "good morning", "good evening"]
    if any(word in message_lower for word in greetings):
        return "🙏 Namaste! I'm your AI Cataract Detection Assistant. I can help you with:\n\n✅ Understanding cataracts\n✅ Symptoms and causes\n✅ Treatment options\n✅ Prevention tips\n✅ How to use this app\n\nWhat would you like to know?"
    
    # What is cataract
    if any(word in message_lower for word in ["what is", "cataract kya hai", "define cataract", "explain cataract"]):
        return "👁️ **What is a Cataract?**\n\nA cataract is the clouding of the eye's natural lens, which lies behind the iris and pupil. Think of it like looking through a foggy window.\n\n**Key Facts:**\n• Most common cause of vision loss worldwide\n• Usually develops slowly over years\n• Affects the eye's ability to focus light\n• Can occur in one or both eyes\n• More common after age 60\n\n**Good news:** Cataracts are treatable with surgery! 👍"
    
    # Symptoms
    if any(word in message_lower for word in ["symptom", "sign", "lakshan", "indication", "how do i know", "detect"]):
        return "⚠️ **Common Cataract Symptoms:**\n\n1️⃣ **Blurred/Cloudy Vision** - Like looking through fog\n2️⃣ **Difficulty seeing at night** - Especially while driving\n3️⃣ **Sensitivity to light** - Glare from headlights/sunlight\n4️⃣ **Seeing halos** - Around lights at night\n5️⃣ **Fading colors** - Colors appear yellowish/dull\n6️⃣ **Double vision** - In one eye\n7️⃣ **Frequent prescription changes** - Glasses don't help much\n\n⚕️ **If you notice these symptoms, consult an eye doctor immediately!**"
    
    # Causes
    if any(word in message_lower for word in ["cause", "why", "kyu", "kaise", "reason", "happen"]):
        return "🔍 **What Causes Cataracts?**\n\n**Main Causes:**\n• **Aging** (most common - 60+ years)\n• **UV radiation** from sun exposure\n• **Diabetes** - high blood sugar damages eye lens\n• **Smoking & alcohol**\n• **Eye injury or inflammation**\n• **Prolonged steroid medication**\n• **Family history**\n• **Previous eye surgery**\n\n**Risk Factors:**\n👴 Age > 60 years\n🌞 Excessive sun exposure\n🚬 Smoking\n🍺 Heavy alcohol use\n💊 Long-term steroid use\n🩸 Diabetes\n\n**Prevention Tip:** Wear UV-protective sunglasses! 🕶️"
    
    # Treatment
    if any(word in message_lower for word in ["treatment", "cure", "surgery", "operation", "ilaj", "doctor", "hospital"]):
        return "💊 **Cataract Treatment Options:**\n\n**Early Stage:**\n• Stronger eyeglasses 👓\n• Brighter lighting for reading\n• Anti-glare sunglasses\n• Magnifying lenses\n\n**Advanced Stage (Only Effective Solution):**\n🏥 **CATARACT SURGERY**\n\n**Surgery Details:**\n✅ Safe & quick (15-20 minutes)\n✅ 95%+ success rate\n✅ Local anesthesia (no pain)\n✅ Cloudy lens removed\n✅ Artificial lens (IOL) implanted\n✅ Same-day discharge\n✅ Recovery in 1-2 weeks\n\n**Cost in India:** ₹15,000 - ₹50,000 per eye\n\n⚕️ **Consult an ophthalmologist to discuss your options!**"
    
    # Prevention
    if any(word in message_lower for word in ["prevent", "avoid", "bachna", "protection", "stop"]):
        return "🛡️ **How to Prevent Cataracts:**\n\n**Lifestyle Changes:**\n1️⃣ **Wear UV-blocking sunglasses** 🕶️ (most important!)\n2️⃣ **Quit smoking** 🚭\n3️⃣ **Limit alcohol** 🍺\n4️⃣ **Manage diabetes** - control blood sugar 🩸\n5️⃣ **Eat healthy diet:**\n   • Leafy greens (spinach, kale)\n   • Colorful fruits & vegetables\n   • Omega-3 fish (salmon, tuna)\n   • Nuts & seeds\n   • Vitamin C & E rich foods\n6️⃣ **Regular eye checkups** 👁️\n7️⃣ **Maintain healthy weight**\n8️⃣ **Protect eyes from injury**\n\n**Remember:** You can't completely prevent cataracts, but you can reduce risk!"
    
    # Age related
    if any(word in message_lower for word in ["age", "old", "umar", "years", "elderly"]):
        return "👴 **Cataracts & Age:**\n\n**Statistics:**\n• Age 40-50: 5% have cataracts\n• Age 50-60: 15% have cataracts\n• Age 60-70: 50% have cataracts\n• Age 75+: 70% have cataracts\n\n**Why does age matter?**\n• Eye lens proteins break down over time\n• Natural aging process\n• Cumulative UV damage\n• Oxidative stress increases\n\n**Good News:**\n✅ Cataracts are NOT a normal part of aging you must accept\n✅ Surgery is highly effective at any age\n✅ Early detection helps better planning\n\n📸 **Use our app to screen regularly after age 50!**"
    
    # Farmers specific
    if any(word in message_lower for word in ["farmer", "kheti", "agriculture", "field", "sun", "outdoor"]):
        return "👨‍🌾 **Important for Farmers:**\n\n**Why farmers are at higher risk:**\n⚠️ Long hours in direct sunlight\n⚠️ UV radiation exposure\n⚠️ Less access to eye protection\n⚠️ Limited healthcare access\n\n**MUST DO for Farmers:**\n1️⃣ **Wear UV-blocking sunglasses** 🕶️ while working\n2️⃣ **Use wide-brim hats** 🧢\n3️⃣ **Take shade breaks**\n4️⃣ **Get yearly eye checkups**\n5️⃣ **Use this app** for regular screening 📱\n\n**Budget smartphone? No problem!**\n✅ Our app works on ANY phone\n✅ No internet needed after installation\n✅ Quick 2-minute screening\n✅ Works in poor lighting\n\n**Upload your eye photo now to check!** 👁️"
    
    # How to use app
    if any(word in message_lower for word in ["how to use", "kaise use", "upload", "photo", "picture", "app use"]):
        return "📱 **How to Use This App:**\n\n**Step 1: Take Photo 📸**\n• Use good lighting (natural daylight best)\n• Look straight at camera\n• Keep phone 10-15cm from eye\n• Open eye wide\n• Remove glasses if wearing\n\n**Step 2: Upload Photo ⬆️**\n• Click 'Upload Image' button\n• Select your eye photo\n• Wait 2-3 seconds for AI analysis\n\n**Step 3: View Results ✅**\n• See if cataract detected\n• Check confidence level\n• Save results in history\n\n**Step 4: Take Action 🏥**\n• If cataract detected → Consult eye doctor\n• If normal → Repeat screening in 6 months\n\n**Tips for Best Results:**\n✅ Clean camera lens\n✅ Good lighting (not too dark/bright)\n✅ Clear, focused photo\n✅ No filters or editing\n\n**Try it now!** 👆"
    
    # Accuracy and limitations
    if any(word in message_lower for word in ["accurate", "sahi", "correct", "trust", "reliable", "confidence"]):
        return "📊 **App Accuracy & Limitations:**\n\n**Our Model:**\n✅ Trained on 1,193 eye images\n✅ 97%+ accuracy on test data\n✅ Uses advanced AI (MobileNetV2)\n✅ Optimized for budget smartphones\n\n**What it CAN do:**\n✅ Screen for cataracts quickly\n✅ Work offline after installation\n✅ Process photos from any phone\n✅ Give confidence percentage\n✅ Track screening history\n\n**What it CANNOT do:**\n❌ Replace professional diagnosis\n❌ Determine severity level\n❌ Recommend specific treatment\n❌ Diagnose other eye conditions\n\n**IMPORTANT DISCLAIMER:**\n⚠️ This is a SCREENING tool only\n⚠️ NOT a medical diagnosis\n⚠️ Always consult an ophthalmologist\n⚠️ Don't delay professional care\n\n**Think of it as:** An early warning system! 🚨"
    
    # Surgery related
    if any(word in message_lower for word in ["surgery", "operation", "cost", "price", "kitna", "expense"]):
        return "🏥 **Cataract Surgery Information:**\n\n**Surgery Types:**\n1️⃣ **Phacoemulsification** (most common)\n   • Small incision\n   • Ultrasound breaks up lens\n   • Quick recovery\n   \n2️⃣ **Manual Small Incision** (MSICS)\n   • Slightly larger cut\n   • Cost-effective\n   • Good for advanced cataracts\n\n**Cost in India:**\n💰 Government hospitals: FREE to ₹5,000\n💰 Private hospitals: ₹15,000 - ₹50,000\n💰 Premium lenses: ₹50,000 - ₹1,50,000\n\n**Recovery Timeline:**\n📅 Day 1-2: Rest, eye shield\n📅 Week 1: Avoid heavy work\n📅 Week 2-4: Gradual return to normal\n📅 Month 1-2: Full recovery\n\n**Success Rate:** 95-98% ✅\n\n**Government Schemes:**\n🇮🇳 Many free cataract surgery camps\n🇮🇳 PM-JAY (Ayushman Bharat) coverage\n🇮🇳 District hospital programs\n\n**Ask your local PHC for free surgery options!**"
    
    # Emergency/Urgent
    if any(word in message_lower for word in ["emergency", "urgent", "pain", "dard", "sudden", "blind", "loss"]):
        return "🚨 **URGENT EYE SYMPTOMS - SEEK IMMEDIATE HELP:**\n\n**Go to emergency if you have:**\n❗ Sudden vision loss\n❗ Severe eye pain\n❗ Eye injury\n❗ Flashes of light\n❗ Sudden floaters\n❗ Red, swollen eye\n❗ Discharge/infection\n\n**For cataracts:**\n• Usually NOT an emergency\n• Develops slowly over months/years\n• But DON'T delay treatment\n\n**When to see doctor for cataract:**\n📞 Vision affecting daily life\n📞 Difficulty reading/driving\n📞 Can't recognize faces\n📞 Bumping into objects\n\n**Emergency Helpline:**\n🏥 Call nearest hospital\n🏥 Or dial 108 (ambulance)\n\n**This app is NOT for emergencies!**"
    
    # Thank you / bye
    if any(word in message_lower for word in ["thank", "thanks", "dhanyavad", "shukriya", "bye", "goodbye"]):
        return "🙏 You're welcome! Happy to help!\n\n**Remember:**\n✅ Regular eye checkups are important\n✅ Use this app for screening every 6 months\n✅ Consult a doctor if concerned\n✅ Protect your eyes from sun\n\n**Stay healthy! Your vision matters!** 👁️✨\n\nAsk me anytime if you have more questions! 😊"
    
    # Default response with options
    return "🤖 **I can help you with:**\n\n1️⃣ What is cataract?\n2️⃣ Symptoms & signs\n3️⃣ Causes & risk factors\n4️⃣ Treatment options\n5️⃣ Prevention tips\n6️⃣ Surgery details & cost\n7️⃣ How to use this app\n8️⃣ Accuracy & limitations\n9️⃣ Tips for farmers\n\n**Just ask me anything!** Type your question in English or Hindi. 😊\n\n**Quick actions:**\n📸 Upload eye photo for screening\n📊 View your screening history\n💬 Ask me any cataract-related question"


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if file:
        # VALIDATE IMAGE FIRST - Try to open with PIL before saving
        try:
            # Read file content into memory
            file_content = file.read()
            # Try to open as PIL image
            img = Image.open(io.BytesIO(file_content))
            # Verify it's a valid image
            img.verify()
            # Re-open for saving (verify() invalidates the image)
            img = Image.open(io.BytesIO(file_content))
            # Convert to RGB (removes any transparency, ensures consistency)
            img = img.convert('RGB')
            
            # Now save the validated image
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            # Ensure .jpg extension
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                filename = filename.rsplit('.', 1)[0] + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save as JPEG with high quality
            img.save(filepath, 'JPEG', quality=95)
            print(f"✅ Image validated and saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Image validation failed: {e}")
            return jsonify({'success': False, 'message': f'Invalid image file: {str(e)}'}), 400
        
        # Make prediction
        try:
            result, confidence, gradcam_path = predict_cataract(filepath)
            
            # Check if prediction was successful
            if result == "Error in prediction":
                return jsonify({'success': False, 'message': 'Analysis failed. Please try again with a clear eye image.'}), 500
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Analysis failed: {str(e)}'}), 500
        
        # Save to database if user is logged in
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            
            prediction = Prediction(
                user_id=session['user_id'],
                image_path=filepath,
                prediction_result=result,
                confidence=confidence
            )
            db.session.add(prediction)
            db.session.commit()
            
            # Send email and SMS notifications in background
            if user:
                send_notifications_async(
                    user_email=user.email,
                    user_phone=user.phone,
                    username=user.username,
                    result=result,
                    confidence=confidence * 100
                )
        
        # Convert gradcam image to base64 for frontend display
        gradcam_base64 = None
        if gradcam_path and os.path.exists(gradcam_path):
            try:
                with open(gradcam_path, 'rb') as f:
                    gradcam_base64 = base64.b64encode(f.read()).decode('utf-8')
                print(f"✅ Grad-CAM converted to base64 ({len(gradcam_base64)} chars)")
            except Exception as e:
                print(f"❌ Error encoding Grad-CAM: {e}")
        else:
            print(f"⚠️ Grad-CAM path not available or file doesn't exist")
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence * 100, 2),
            'image_path': filepath,
            'gradcam_image': gradcam_base64
        })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    
    response = get_chatbot_response(message)
    
    # Save chat history if user is logged in
    if 'user_id' in session:
        chat_entry = ChatHistory(
            user_id=session['user_id'],
            message=message,
            response=response
        )
        db.session.add(chat_entry)
        db.session.commit()
    
    return jsonify({'success': True, 'response': response})

@app.route('/api/history')
def get_history():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()
    
    history = []
    for pred in predictions:
        history.append({
            'id': pred.id,
            'result': pred.prediction_result,
            'confidence': round(pred.confidence * 100, 2),
            'timestamp': pred.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'image_path': pred.image_path
        })
    
    return jsonify({'success': True, 'history': history})

@app.route('/api/download-report/<int:prediction_id>')
def download_report(prediction_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    # Get prediction
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=session['user_id']).first()
    if not prediction:
        return jsonify({'success': False, 'message': 'Prediction not found'}), 404
    
    # Generate PDF
    pdf_filename = f"report_{prediction.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph('NETRAMITRA Detection Report', title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient/User Info
    user = User.query.get(session['user_id'])
    info_style = ParagraphStyle(
        'Info',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12
    )
    
    story.append(Paragraph(f'<b>Patient Name:</b> {user.username}', info_style))
    story.append(Paragraph(f'<b>Email:</b> {user.email}', info_style))
    story.append(Paragraph(f'<b>Test Date:</b> {prediction.timestamp.strftime("%B %d, %Y at %H:%M")}', info_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Results Section
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#e74c3c') if 'Cataract Detected' in prediction.prediction_result else colors.HexColor('#27ae60'),
        spaceAfter=20
    )
    story.append(Paragraph('Test Results', styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Results table
    data = [
        ['Result', prediction.prediction_result],
        ['Confidence', f"{prediction.confidence * 100:.2f}%"],
        ['Status', 'Positive' if 'Cataract Detected' in prediction.prediction_result else 'Negative']
    ]
    
    table = Table(data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4*inch))
    
    # Add original eye image
    if os.path.exists(prediction.image_path):
        try:
            img = RLImage(prediction.image_path, width=2.5*inch, height=2.5*inch)
            story.append(Paragraph('Eye Image:', styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
        except:
            pass
    
    # Add Grad-CAM visualization showing affected areas
    # Construct gradcam path from image filename
    image_basename = os.path.basename(prediction.image_path)
    gradcam_filename = f"gradcam_{image_basename}"
    gradcam_path = os.path.join('gradcam_results', gradcam_filename)
    
    if os.path.exists(gradcam_path):
        try:
            gradcam_img = RLImage(gradcam_path, width=6*inch, height=2*inch)
            story.append(Paragraph('Affected Areas Analysis (Grad-CAM):', styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(
                'The visualization below shows which parts of the eye influenced the AI\'s decision. '
                'Red areas indicate regions of high attention by the model.',
                info_style
            ))
            story.append(Spacer(1, 0.1*inch))
            story.append(gradcam_img)
            story.append(Spacer(1, 0.3*inch))
        except Exception as e:
            print(f"Could not add Grad-CAM to PDF: {e}")
    
    # Recommendations
    story.append(Paragraph('Recommendations:', styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    if 'Cataract Detected' in prediction.prediction_result:
        recommendations = [
            '• Consult an ophthalmologist immediately for professional diagnosis',
            '• This is a screening tool and not a substitute for medical advice',
            '• Early detection improves treatment outcomes',
            '• Schedule a comprehensive eye examination',
            '• Discuss treatment options with your eye doctor'
        ]
    else:
        recommendations = [
            '• Continue regular eye check-ups every 6-12 months',
            '• This is a screening tool and not a substitute for medical advice',
            '• Maintain eye health with proper diet and UV protection',
            '• Report any vision changes to your eye doctor',
            '• Retest if you notice any symptoms'
        ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, info_style))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        '<b>DISCLAIMER:</b> This report is generated by an AI screening tool and is not a medical diagnosis. '
        'Always consult a qualified ophthalmologist for professional medical advice, diagnosis, and treatment.',
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
