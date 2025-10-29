import os
import re
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_PATH = r"C:\Users\Varsha Krishnan\Documents\projects\temporal\data\Annotated_data_part1"
OUTPUT_PATH = r"C:\Users\Varsha Krishnan\Documents\projects\temporal\data\processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)
TARGET_SIZE = (224, 224)

print("="*80)
print("MINTPAIN DATA PREPROCESSING - FIXED STRUCTURE DETECTION")
print("="*80)
print(f"Dataset Path: {DATASET_PATH}")
print(f"Output Path : {OUTPUT_PATH}")
print("-"*80)

# =====================================================
# HELPERS
# =====================================================

def extract_label(folder_name):
    """Extracts numeric pain label (Label0–Label4) from folder name."""
    match = re.search(r'Label(\d+)', folder_name, re.IGNORECASE)
    return int(match.group(1)) if match else None

def normalize_frames(rgb, depth, thermal):
    """Normalizes frames for each modality."""
    rgb_norm = rgb.astype(np.float32) / 255.0
    depth_norm = (depth.astype(np.float32) - np.mean(depth)) / (np.std(depth) + 1e-6)
    tmin, tmax = np.percentile(thermal, [2, 98])
    thermal_norm = np.clip((thermal.astype(np.float32) - tmin) / (tmax - tmin + 1e-6), 0, 1)
    return rgb_norm, depth_norm[..., None], thermal_norm[..., None]

def center_crop(rgb_frame, depth_frame, thermal_frame):
    """
    Crops fixed center region across all modalities with better validation.
    """
    try:
        # Validate input shapes
        if rgb_frame.shape[:2] != depth_frame.shape or rgb_frame.shape[:2] != thermal_frame.shape:
            print(f"Shape mismatch: RGB={rgb_frame.shape}, D={depth_frame.shape}, T={thermal_frame.shape}")
            return None, None, None

        h, w = rgb_frame.shape[:2]
        
        # Use smaller crop size for stability
        crop_size = min(h, w) // 4  # Changed from 2 to 4 for smaller crop
        
        # Calculate center coordinates
        cx, cy = w // 2, h // 2
        
        # Calculate crop boundaries with validation
        x1 = max(0, cx - crop_size)
        y1 = max(0, cy - crop_size)
        x2 = min(w, cx + crop_size)
        y2 = min(h, cy + crop_size)
        
        # Validate crop region size
        if (x2 - x1) < 10 or (y2 - y1) < 10:  # Minimum size check
            print(f"Crop region too small: {x2-x1}x{y2-y1}")
            return None, None, None

        # Perform crops
        rgb_face = rgb_frame[y1:y2, x1:x2]
        depth_face = depth_frame[y1:y2, x1:x2]
        thermal_face = thermal_frame[y1:y2, x1:x2]

        # Validate crops
        if any(x.size == 0 for x in [rgb_face, depth_face, thermal_face]):
            print("Empty crop detected")
            return None, None, None

        # Resize with validation
        rgb_face = cv2.resize(rgb_face, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        depth_face = cv2.resize(depth_face, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        thermal_face = cv2.resize(thermal_face, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

        return rgb_face, depth_face, thermal_face

    except Exception as e:
        print(f"Cropping error: {str(e)}")
        return None, None, None

# Add this debug function after the helpers section
def debug_folder_structure(sweep_dir):
    """Debug helper to check folder structure and files"""
    print(f"\nDEBUG: Checking {sweep_dir.name}")
    
    # Check modality folders
    d_dir = sweep_dir / 'D'
    rgb_dir = sweep_dir / 'RGB'
    t_dir = sweep_dir / 'T'
    
    print(f"  D folder exists: {d_dir.exists()}")
    print(f"  RGB folder exists: {rgb_dir.exists()}")
    print(f"  T folder exists: {t_dir.exists()}")
    
    if all(d.exists() for d in [d_dir, rgb_dir, t_dir]):
        # Count image files
        rgb_files = list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg'))
        depth_files = list(d_dir.glob('*.png')) + list(d_dir.glob('*.jpg'))
        thermal_files = list(t_dir.glob('*.png')) + list(t_dir.glob('*.jpg'))
        
        print(f"  RGB files found: {len(rgb_files)}")
        print(f"  Depth files found: {len(depth_files)}")
        print(f"  Thermal files found: {len(thermal_files)}")
        
        if rgb_files:
            print(f"  Sample RGB file: {rgb_files[0].name}")
            img = cv2.imread(str(rgb_files[0]))
            if img is not None:
                print(f"  Sample image shape: {img.shape}")
            else:
                print("  ❌ Failed to read sample image!")

# =====================================================
# MAIN PIPELINE
# =====================================================

all_rgb, all_depth, all_thermal, all_labels = [], [], [], []

subjects = [s for s in Path(DATASET_PATH).iterdir() if s.is_dir() and 'sub' in s.name.lower()]
if not subjects:
    raise RuntimeError("❌ No subject folders found under Annotated_data_part1!")

for sid, subject in enumerate(subjects, 1):
    print(f"\n[{sid}/{len(subjects)}] Processing: {subject.name}")

    # Each subject has trial dirs
    trials = [t for t in subject.iterdir() if t.is_dir()]
    for trial in trials:
        print(f"  Trial: {trial.name}")
        
        # Each trial contains multiple sweep folders (Label0, Label1, etc.)
        sweeps = [s for s in trial.iterdir() if s.is_dir() and 'label' in s.name.lower()]
        print(f"    Found {len(sweeps)} sweep folders")
        
        for sweep in tqdm(sweeps, desc=f"    Processing sweeps", leave=True):
            label = extract_label(sweep.name)
            if label is None:
                print(f"    Skipping {sweep.name}: no label found")
                continue
                
            # Add debug info
            debug_folder_structure(sweep)
            
            d_dir = sweep / 'D'
            rgb_dir = sweep / 'RGB'
            t_dir = sweep / 'T'

            if not (d_dir.exists() and rgb_dir.exists() and t_dir.exists()):
                print(f"    Skipping {sweep.name}: missing modality folders")
                continue

            rgb_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
            depth_files = sorted(list(d_dir.glob('*.png')) + list(d_dir.glob('*.jpg')))
            thermal_files = sorted(list(t_dir.glob('*.png')) + list(t_dir.glob('*.jpg')))
            
            min_count = min(len(rgb_files), len(depth_files), len(thermal_files))
            if min_count == 0:
                print(f"    Skipping {sweep.name}: no matching files found")
                continue

            frames_ok = 0
            for i in range(min_count):
                try:
                    rgb = cv2.imread(str(rgb_files[i]))
                    depth = cv2.imread(str(depth_files[i]), cv2.IMREAD_GRAYSCALE)
                    thermal = cv2.imread(str(thermal_files[i]), cv2.IMREAD_GRAYSCALE)
                    
                    if rgb is None or depth is None or thermal is None:
                        print(f"    Failed to read frame {i}")
                        continue

                    rgb_crop, depth_crop, thermal_crop = center_crop(rgb, depth, thermal)
                    if rgb_crop is None:
                        print(f"    Failed to crop frame {i}")
                        continue

                    rgb_norm, depth_norm, thermal_norm = normalize_frames(rgb_crop, depth_crop, thermal_crop)
                    all_rgb.append(rgb_norm)
                    all_depth.append(depth_norm)
                    all_thermal.append(thermal_norm)
                    all_labels.append(label)
                    frames_ok += 1
                    
                except Exception as e:
                    print(f"    Error processing frame {i}: {str(e)}")
                    continue

            if frames_ok > 0:
                print(f"    ✓ {sweep.name}: {frames_ok}/{min_count} frames processed (Label={label})")
            else:
                print(f"    ❌ {sweep.name}: No frames successfully processed")

# =====================================================
# SAVE OUTPUT
# =====================================================

print("\n" + "="*80)
print(f"Samples collected: {len(all_labels)}")
if len(all_labels) == 0:
    raise RuntimeError("❌ No valid images detected (check .PNG/.JPG availability).")

all_rgb, all_depth, all_thermal, all_labels = (
    np.array(all_rgb), np.array(all_depth), np.array(all_thermal), np.array(all_labels)
)

train_rgb, val_rgb, train_depth, val_depth, train_thermal, val_thermal, train_labels, val_labels = train_test_split(
    all_rgb, all_depth, all_thermal, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

np.savez(os.path.join(OUTPUT_PATH, 'train_data.npz'),
         rgb=train_rgb, depth=train_depth, thermal=train_thermal, labels=train_labels)
np.savez(os.path.join(OUTPUT_PATH, 'val_data.npz'),
         rgb=val_rgb, depth=val_depth, thermal=val_thermal, labels=val_labels)

print(f"\n✓ Data saved to: {OUTPUT_PATH}")
print(f"  Train samples: {len(train_labels)} | Val samples: {len(val_labels)}")
print("="*80)
print("You can now run: python train.py")
print("="*80)
