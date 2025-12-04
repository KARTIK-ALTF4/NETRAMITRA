@echo off
echo ========================================
echo GitHub Desktop Upload Instructions
echo ========================================
echo.
echo Since Git is not installed, use GitHub Desktop (easier):
echo.
echo 1. Download GitHub Desktop:
echo    https://desktop.github.com/
echo.
echo 2. Install and sign in with your GitHub account
echo.
echo 3. Click "Add" - "Add Existing Repository"
echo.
echo 4. Browse to: %CD%
echo.
echo 5. Click "Publish repository"
echo.
echo 6. Name: cataract-detection-ai
echo.
echo 7. Description: AI-powered cataract detection with EfficientNet-B0
echo.
echo 8. Uncheck "Keep this code private" (for public repo)
echo.
echo 9. Click "Publish Repository"
echo.
echo Done! Your project will be on GitHub!
echo.
echo ========================================
echo Model File (53.85 MB) - Upload Separately
echo ========================================
echo.
echo The trained model is too large for direct upload.
echo After publishing, go to your repo on GitHub:
echo.
echo 1. Click "Releases" - "Create a new release"
echo 2. Tag: v1.0
echo 3. Title: "Trained Model - EfficientNet-B0"
echo 4. Drag and drop: pytorch_checkpoints\best_model.pth
echo 5. Click "Publish release"
echo.
pause
