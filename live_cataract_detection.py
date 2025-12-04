"""
Real-Time Cataract Detection with MediaPipe Eye Cropping
Integrates with existing PyTorch model for live detection
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import mediapipe as mp
import torchvision.transforms as transforms
from model import create_model
import time


class RealTimeCataractDetector:
    """
    Real-time cataract detection using webcam + MediaPipe eye detection
    """
    
    def __init__(self, model_path='pytorch_checkpoints/best_model.pth'):
        """Initialize model and MediaPipe"""
        
        # Load PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = create_model(num_classes=2, dropout_rate=0.5, freeze_backbone=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Initialize MediaPipe Face Mesh (for eye detection)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe eye landmark indices
        # Left eye: 33, 133, 160, 144, 158, 153
        # Right eye: 362, 263, 387, 373, 380, 385
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Image preprocessing for model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Detection history for smoothing
        self.detection_history = []
        self.history_size = 5
        
        print("✅ MediaPipe initialized!")
    
    def get_eye_landmarks(self, face_landmarks, image_shape, eye_indices):
        """Extract eye landmarks and compute bounding box"""
        h, w = image_shape[:2]
        
        points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        points = np.array(points)
        
        # Compute bounding box with padding
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding (30% on each side)
        width = x_max - x_min
        height = y_max - y_min
        padding = max(width, height) * 0.3
        
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(w, int(x_max + padding))
        y_max = min(h, int(y_max + padding))
        
        return (x_min, y_min, x_max, y_max), points
    
    def crop_eye_region(self, image, bbox):
        """Crop eye region from image"""
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Convert to square (required for model)
        h, w = cropped.shape[:2]
        size = max(h, w)
        
        # Create square canvas
        square = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        
        return square
    
    def predict_cataract(self, eye_image):
        """Run cataract detection on cropped eye image"""
        
        # Convert to PIL and preprocess
        pil_image = Image.fromarray(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        # Get predictions
        cataract_prob = probabilities[0].item()
        normal_prob = probabilities[1].item()
        
        prediction = 'Normal Eye' if normal_prob > cataract_prob else 'Possible Cataract Detected'
        confidence = max(cataract_prob, normal_prob) * 100
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'cataract_probability': cataract_prob * 100,
            'normal_probability': normal_prob * 100
        }
    
    def smooth_predictions(self, result):
        """Smooth predictions over time to reduce flickering"""
        self.detection_history.append(result)
        
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Average probabilities
        avg_cataract = np.mean([r['cataract_probability'] for r in self.detection_history])
        avg_normal = np.mean([r['normal_probability'] for r in self.detection_history])
        
        prediction = 'Normal Eye' if avg_normal > avg_cataract else 'Possible Cataract Detected'
        confidence = max(avg_cataract, avg_normal)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'cataract_probability': avg_cataract,
            'normal_probability': avg_normal
        }
    
    def draw_results(self, frame, eye_bbox, result, eye_type):
        """Draw detection results on frame"""
        x_min, y_min, x_max, y_max = eye_bbox
        
        # Draw bounding box around eye
        color = (0, 0, 255) if result['prediction'] == 'Possible Cataract Detected' else (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"{eye_type}: {result['prediction']}"
        conf_text = f"Confidence: {result['confidence']:.1f}%"
        
        # Background for text
        (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (w2, h2), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(frame, (x_min, y_min - h1 - h2 - 20), (x_min + max(w1, w2) + 10, y_min), color, -1)
        
        # Text
        cv2.putText(frame, label, (x_min + 5, y_min - h2 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, conf_text, (x_min + 5, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_live_detection(self, camera_index=0, detect_both_eyes=True):
        """
        Run live cataract detection from webcam
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            detect_both_eyes: If True, detect both eyes; if False, detect best visible eye
        """
        
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("\n" + "="*80)
        print("REAL-TIME CATARACT DETECTION ACTIVE")
        print("="*80)
        print("Instructions:")
        print("  • Position your eye in front of the camera")
        print("  • Keep eye open and look at camera")
        print("  • Wait for green/red box around eye")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save screenshot")
        print("  • Press 'c' to capture and analyze")
        print("="*80 + "\n")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Detect and analyze eyes
                eyes_detected = []
                
                # Right eye (appears on left in mirrored view)
                right_bbox, right_points = self.get_eye_landmarks(
                    face_landmarks, frame.shape, self.RIGHT_EYE_INDICES
                )
                right_eye_crop = self.crop_eye_region(frame, right_bbox)
                right_result = self.predict_cataract(right_eye_crop)
                right_result_smoothed = self.smooth_predictions(right_result)
                
                frame = self.draw_results(frame, right_bbox, right_result_smoothed, "Right Eye")
                eyes_detected.append(('Right', right_result_smoothed))
                
                if detect_both_eyes:
                    # Left eye
                    left_bbox, left_points = self.get_eye_landmarks(
                        face_landmarks, frame.shape, self.LEFT_EYE_INDICES
                    )
                    left_eye_crop = self.crop_eye_region(frame, left_bbox)
                    left_result = self.predict_cataract(left_eye_crop)
                    left_result_smoothed = self.smooth_predictions(left_result)
                    
                    frame = self.draw_results(frame, left_bbox, left_result_smoothed, "Left Eye")
                    eyes_detected.append(('Left', left_result_smoothed))
                
                # Display summary
                y_pos = 30
                for eye_type, result in eyes_detected:
                    color = (0, 0, 255) if result['prediction'] == 'Possible Cataract Detected' else (0, 255, 0)
                    text = f"{eye_type}: {result['prediction']} ({result['confidence']:.1f}%)"
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, color, 2)
                    y_pos += 30
            else:
                # No face detected
                cv2.putText(frame, "No face detected - Please look at camera", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time > 1:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Display FPS and device info
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Device: {self.device}", (frame.shape[1] - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Real-Time Cataract Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✅ Detection stopped by user")
                break
            elif key == ord('s'):
                filename = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('c'):
                print("\n" + "="*80)
                print("CAPTURE & DETAILED ANALYSIS")
                print("="*80)
                if results.multi_face_landmarks:
                    for eye_type, result in eyes_detected:
                        print(f"\n{eye_type} Eye:")
                        print(f"  Prediction: {result['prediction']}")
                        print(f"  Confidence: {result['confidence']:.2f}%")
                        print(f"  Cataract Probability: {result['cataract_probability']:.2f}%")
                        print(f"  Normal Probability: {result['normal_probability']:.2f}%")
                print("="*80 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")


def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("REAL-TIME CATARACT DETECTION SYSTEM")
    print("="*80)
    print("\nThis system:")
    print("  1. Uses MediaPipe to detect eyes in real-time")
    print("  2. Automatically crops just the eye region")
    print("  3. Sends cropped image to cataract detection model")
    print("  4. Returns: 'Possible Cataract Detected' or 'Normal Eye' + confidence")
    print("\n✅ This flow SOLVES the eyebrow spurious correlation issue!")
    print("   (MediaPipe crops remove eyebrows automatically)")
    print("="*80 + "\n")
    
    # Check if MediaPipe is installed
    try:
        import mediapipe
        print("✅ MediaPipe installed")
    except ImportError:
        print("❌ MediaPipe not installed!")
        print("\nInstall with: pip install mediapipe")
        return
    
    # Initialize detector
    detector = RealTimeCataractDetector()
    
    # Run live detection
    print("\nStarting camera...")
    detector.run_live_detection(camera_index=0, detect_both_eyes=True)


if __name__ == '__main__':
    main()
