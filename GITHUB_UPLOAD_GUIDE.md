# GitHub Upload Guide - Cataract Detection Project

## Prerequisites

1. **Install Git**
   - Download: https://git-scm.com/download/win
   - Run installer and restart terminal after installation

2. **Create GitHub Account** (if not already)
   - Go to https://github.com
   - Sign up / Login

## Step 1: Update .gitignore

Your `.gitignore` file already exists. Make sure it includes:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Flask
instance/
.webassets-cache

# PyTorch
*.pth
*.pt
pytorch_checkpoints/
pytorch_runs/

# Data
data/
uploads/
uploads_cropped/
reports/
models/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Model files
*.h5
*.caffemodel
deploy.prototxt
res10_300x300_ssd_iter_140000_fp16.caffemodel

# Results
gradcam_results/
model_documentation/
confusion_matrix.png
model_comparison.png
model_comparison_full.png
```

## Step 2: Initialize Git (After Installing)

Open PowerShell in project directory:

```powershell
cd 'c:\Users\Harshita\Downloads\Kartik cataract\cataract detection'

# Initialize git
git init

# Configure git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `cataract-detection-ai`
3. Description: `AI-powered cataract detection system using EfficientNet-B0 with Grad-CAM explainability`
4. Choose: **Public** (or Private)
5. **DO NOT** initialize with README (we already have one)
6. Click **Create repository**

## Step 4: Add Files and Commit

```powershell
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# Create first commit
git commit -m "Initial commit: Cataract detection system with EfficientNet-B0"
```

## Step 5: Connect to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/cataract-detection-ai.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 6: Upload Model File Separately

Since `best_model.pth` is 53.85 MB, you have options:

### Option A: Git LFS (Large File Storage)
```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git add .gitattributes
git add pytorch_checkpoints/best_model.pth
git commit -m "Add trained model with Git LFS"
git push
```

### Option B: Google Drive / Dropbox
1. Upload `pytorch_checkpoints/best_model.pth` to Google Drive
2. Get shareable link
3. Add link to README.md:

```markdown
## Download Trained Model

Due to file size, the trained model is hosted separately:
- **Download**: [best_model.pth (53.85 MB)](YOUR_GOOGLE_DRIVE_LINK)
- Place in: `pytorch_checkpoints/best_model.pth`
```

### Option C: GitHub Releases
1. Go to your repo → Releases → Create new release
2. Tag: `v1.0`
3. Title: `Cataract Detection v1.0 - Trained Model`
4. Attach `best_model.pth` file
5. Publish release

## Step 7: Update README with Setup Instructions

Your README.md should include:

```markdown
# 🩺 Cataract Detection AI System

AI-powered cataract detection using EfficientNet-B0 with Grad-CAM explainability.

## 🎯 Features
- ✅ 98-100% validation accuracy
- ✅ Grad-CAM visualization showing affected areas
- ✅ PDF report generation
- ✅ Patient history tracking
- ✅ RESTful API endpoints

## 🚀 Quick Start

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/cataract-detection-ai.git
cd cataract-detection-ai
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Download Model
Download trained model from [Releases](link) and place in:
\`\`\`
pytorch_checkpoints/best_model.pth
\`\`\`

### 4. Run Application
\`\`\`bash
python app.py
\`\`\`

Open browser: http://localhost:5000

## 📊 Model Performance
- **Architecture**: EfficientNet-B0
- **Parameters**: 4.66M
- **Best Val Accuracy**: 100%
- **Training**: 18 epochs

## 📁 Project Structure
\`\`\`
cataract-detection/
├── app.py                    # Flask web application
├── model.py                  # EfficientNet-B0 model
├── train.py                  # Training script
├── dataloader.py             # Data loading
├── gradcam_visualization.py  # Explainability
├── templates/                # HTML templates
├── static/                   # CSS/JS/Images
├── pytorch_checkpoints/      # Trained models
└── requirements.txt          # Dependencies
\`\`\`

## 🔬 Technology Stack
- **Backend**: Flask, PyTorch
- **Model**: EfficientNet-B0 (torchvision)
- **Explainability**: Grad-CAM
- **Database**: SQLite
- **Reports**: ReportLab

## 📖 Documentation
- [Model Comparison Report](MODEL_COMPARISON_REPORT.md)
- [Grad-CAM Guide](GRADCAM_GUIDE.md)
- [Quick Start](QUICKSTART.md)

## 📄 License
MIT License
\`\`\`

## Alternative: One-Click Commands

After installing Git, run these commands:

```powershell
cd 'c:\Users\Harshita\Downloads\Kartik cataract\cataract detection'

# Setup
git init
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Commit
git add .
git commit -m "Initial commit: Cataract detection system"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/cataract-detection-ai.git
git branch -M main
git push -u origin main
```

## Troubleshooting

**Error: "Permission denied (publickey)"**
```powershell
# Use HTTPS instead of SSH
git remote set-url origin https://github.com/YOUR_USERNAME/cataract-detection-ai.git
```

**Error: "Large files detected"**
```powershell
# Remove from staging
git rm --cached pytorch_checkpoints/best_model.pth
# Add to .gitignore and use Git LFS or external hosting
```

**Error: "Updates were rejected"**
```powershell
# Force push (only for initial setup)
git push -f origin main
```

## Next Steps After Upload

1. ✅ Add repository description and topics on GitHub
2. ✅ Enable GitHub Actions for CI/CD (optional)
3. ✅ Add LICENSE file
4. ✅ Create GitHub Pages for documentation
5. ✅ Add badges to README (build status, license)
6. ✅ Star your own repo!

---

**Need Help?** Check GitHub documentation: https://docs.github.com
