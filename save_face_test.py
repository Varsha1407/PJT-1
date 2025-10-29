import numpy as np
import cv2
from pathlib import Path
import re
from sklearn.model_selection import train_test_split

BASE_PATH = r'C:\Users\Varsha Krishnan\Documents\projects\temporal\data'
IMG_SIZE = (64, 64)
SEQ_LEN = 12
NUM_CLASSES = 5
MAX_PER_CLASS = 450

def cache_sequences(base_path):
    seqs_by_class = {i: [] for i in range(NUM_CLASSES)}
    for part_folder in Path(base_path).iterdir():
        if not part_folder.is_dir():
            continue
        for sweep in part_folder.rglob("*Label*"):
            if not sweep.is_dir():
                continue
            m = re.search(r'Label(\d+)', sweep.name)
            if not m:
                continue
            label = int(m.group(1))
            if label >= NUM_CLASSES:
                continue  # skip invalid label
            if len(seqs_by_class[label]) >= MAX_PER_CLASS:
                continue
            rgb_dir = sweep / 'RGB'
            if not rgb_dir.exists():
                continue
            imgs = sorted(rgb_dir.glob('*.jpg'))
            for i in range(0, len(imgs) - SEQ_LEN + 1, SEQ_LEN):
                seq = []
                for img_path in imgs[i:i+SEQ_LEN]:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
                    seq.append(img)
                seqs_by_class[label].append(np.stack(seq))
                if len(seqs_by_class[label]) >= MAX_PER_CLASS:
                    break
    X, y = [], []
    for lbl, seqs in seqs_by_class.items():
        for arr in seqs:
            X.append(arr)
            y.append(lbl)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)

X, y = cache_sequences(BASE_PATH)
print('Data loaded:', X.shape, y.shape)

# Train-validation split (same as in your training code)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save test arrays to .npy for fusion.py
np.save('X_test_face.npy', X_test)
np.save('y_test_face.npy', y_test)

print("Test arrays saved for fusion.py.")
