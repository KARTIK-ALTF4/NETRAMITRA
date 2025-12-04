# 👁️ AI-Powered Cataract Detection System

A complete web-based cataract detection system using Deep Learning (EfficientNet-B0) with Grad-CAM visualization, built with Flask and PyTorch.

**Author:** Kartik Koul

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Real-time Cataract Detection** - Upload eye images for instant AI analysis
- **98-100% Accuracy** - EfficientNet-B0 model with transfer learning
- **Grad-CAM Visualization** - Explainable AI showing affected regions
- **PDF Reports** - Download detailed medical reports
- **Patient History** - Track previous scans and results

### Advanced Features
- **Eye Cropping** - Automatic eye region detection using Haar Cascades
- **Email Notifications** - Automatic result delivery via email
- **SMS Alerts** - Optional SMS notifications via Twilio
- **Bilingual Support** - English + Hindi interface
- **Mobile Responsive** - Works on all devices
- **Chat Interface** - AI chatbot for cataract queries

## 🖼️ Screenshots

### Main Interface
- Upload image or use live camera
- Instant AI analysis with confidence score
- 3-panel Grad-CAM visualization (Original | Heatmap | Overlay)

### PDF Report
- Patient information
- Test results with confidence
- Eye image and Grad-CAM visualization
- Personalized recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/cataract-detection.git
cd cataract-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the trained model**
- Place `best_model.pth` in `pytorch_checkpoints/` folder
- Model: EfficientNet-B0 (4.6M parameters)

4. **Initialize database**
```bash
python migrate_database.py
```

5. **Run the application**
```bash
python app.py
```

6. **Open browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
cataract-detection/
├── app.py                          # Main Flask application
├── model.py                        # PyTorch model definition
├── dataloader.py                   # Dataset loader and augmentations
├── train_pytorch.py                # Training script
├── gradcam_visualization.py        # Grad-CAM implementation
├── live_cataract_detection.py      # Real-time webcam detection
├── migrate_database.py             # Database migration script
├── requirements.txt                # Python dependencies
│
├── templates/
│   └── index.html                  # Main web interface
│
├── static/
│   ├── css/
│   │   └── style.css              # Styling
│   └── js/
│       └── script.js              # Frontend logic
│
├── pytorch_checkpoints/
│   └── best_model.pth             # Trained model (download separately)
│
├── instance/
│   └── cataract_detection.db      # SQLite database (auto-created)
│
└── uploads/                        # User uploaded images (auto-created)
```

## 🧠 Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Fine-tuning**: Full network trainable
- **Input Size**: 224×224 RGB images
- **Output**: Binary classification (Cataract / Normal)
- **Regularization**: Dropout (0.5, 0.25)
- **Optimizer**: Adam (LR: 1e-4)
- **Loss**: Cross-Entropy with class weights

### Training Details
- **Dataset**: 160 cataract + 160 normal eye images
- **Splits**: 70% train, 15% validation, 15% test
- **Augmentation**: Rotation, flip, color jitter, affine, blur
- **Early Stopping**: Patience 7 epochs
- **Scheduler**: ReduceLROnPlateau

## 📊 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.5% |
| Precision | 96.8% |
| Recall | 98.2% |
| F1-Score | 97.5% |

## 🔬 How to Use

### 1. Register/Login
- Create an account with email and phone number
- Secure password authentication

### 2. Upload Image
- Click "Upload Photo" or use "Live Camera"
- Ensure clear, well-lit eye image
- Remove glasses if wearing

### 3. View Results
- AI prediction (Cataract Detected / No Cataract)
- Confidence percentage
- **Grad-CAM visualization** showing affected areas
- Personalized recommendations

### 4. Download Report
- Go to "History" section
- Click "Download PDF" button
- Professional medical report with all details

## 🛠️ Configuration

### Email Notifications (Optional)
Edit `app.py`:
```python
EMAIL_USER = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_app_password'  # Generate from Google Account
```

### SMS Notifications (Optional)
Edit `app.py`:
```python
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE = '+1234567890'
```

See `NOTIFICATION_SETUP.md` for detailed setup instructions.

## 📦 Dependencies

Main libraries:
- `torch` - PyTorch deep learning framework
- `torchvision` - Image transformations
- `flask` - Web framework
- `flask-sqlalchemy` - Database ORM
- `opencv-python` - Image processing
- `pillow` - Image handling
- `reportlab` - PDF generation
- `matplotlib` - Visualizations
- `numpy` - Numerical operations
- `twilio` - SMS notifications (optional)

Full list in `requirements.txt`

## 🧪 Testing

### Test Single Image
```bash
python test_any_image.py path/to/eye_image.jpg
```

### Generate Grad-CAM
```bash
python gradcam_visualization.py
```

### Live Webcam Detection
```bash
python live_cataract_detection.py
```

### Evaluate on Test Set
```bash
python evaluate_samples.py
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for screening purposes only and is NOT a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist for:
- Clinical examination
- Confirmed diagnosis
- Treatment recommendations
- Surgical decisions

False positives and false negatives can occur. Do not rely solely on this tool for medical decisions.

## 👨‍💻 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- EfficientNet architecture by Google Brain
- Grad-CAM implementation based on original paper
- Dataset contributors
- Flask and PyTorch communities

## 📧 Contact

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## ⚠️ Disclaimer

For educational purposes only. Not for medical diagnosis. Consult healthcare professionals.

---

**Star ⭐ this repo if helpful!**
