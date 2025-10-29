import os
import cv2
import numpy as np

# Set TF env vars before importing any TF-backed package
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Global detector
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()  # ✅ FIXED: Removed min_face_size
    return _detector

def detect_and_crop_face(rgb_frame, depth_frame, thermal_frame, target_size=(224, 224)):
    """Detect face in RGB frame and crop all modalities accordingly"""
    try:
        # Input validation
        if rgb_frame is None or depth_frame is None or thermal_frame is None:
            return None
            
        if rgb_frame.size == 0 or depth_frame.size == 0 or thermal_frame.size == 0:
            return None
            
        # Ensure RGB frame is in correct format
        if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
            return None
        
        # Check if image is too dark
        if np.mean(rgb_frame) < 10:
            return None
        
        detector = get_detector()
        
        # Detect faces
        faces = detector.detect_faces(rgb_frame)
        
        if not faces:
            return None
        
        # Get largest face
        face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = face['box']
        
        # Filter out faces that are too small
        if w < 50 or h < 50:
            return None
        
        # Add padding
        padding = 20
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(rgb_frame.shape[1], x + w + padding), min(rgb_frame.shape[0], y + h + padding)
        
        # Crop faces from all modalities
        rgb_face = rgb_frame[y1:y2, x1:x2]
        depth_face = depth_frame[y1:y2, x1:x2]
        thermal_face = thermal_frame[y1:y2, x1:x2]
        
        # Validate crops
        if rgb_face.size == 0 or depth_face.size == 0 or thermal_face.size == 0:
            return None
        
        # Resize to uniform dimensions
        rgb_face = cv2.resize(rgb_face, target_size)
        depth_face = cv2.resize(depth_face, target_size)
        thermal_face = cv2.resize(thermal_face, target_size)
        
        # Return homography info
        homography = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        
        return rgb_face, depth_face, thermal_face, homography
        
    except Exception as e:
        print(f"Error in detect_and_crop_face: {str(e)}")
        return None

def preprocess_frames(rgb_face, depth_face, thermal_face):
    """Normalize each modality independently"""
    # RGB: normalize to [0, 1]
    rgb_normalized = rgb_face.astype(np.float32) / 255.0
    
    # Depth: normalize using z-score
    depth_normalized = (depth_face.astype(np.float32) - np.mean(depth_face)) / (np.std(depth_face) + 1e-6)
    
    # Thermal: normalize to [0, 1]
    thermal_min, thermal_max = np.percentile(thermal_face, [2, 98])
    thermal_normalized = np.clip((thermal_face.astype(np.float32) - thermal_min) / (thermal_max - thermal_min + 1e-6), 0, 1)
    
    return rgb_normalized, depth_normalized, thermal_normalized
