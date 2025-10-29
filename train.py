import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import re

# ---- CONFIG ----
BASE_PATH = r'C:\Users\Varsha Krishnan\Documents\projects\temporal\data'
IMG_SIZE = (64, 64)
SEQ_LEN = 12
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 5
MAX_PER_CLASS = 450  # set to fit RAM

# ---- BUILD FILE LIST AND CACHE ----
def cache_sequences(base_path):
    seqs_by_class = {i: [] for i in range(NUM_CLASSES)}
    for part_folder in Path(base_path).iterdir():
        if not part_folder.is_dir():
            continue
        print(f"Scanning: {part_folder.name}")
        for sweep in part_folder.rglob("*Label*"):
            if not sweep.is_dir():
                continue
            m = re.search(r'Label(\d+)', sweep.name)
            if not m:
                continue
            label = int(m.group(1))
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
    # flatten to X, y for all classes
    X, y = [], []
    for lbl, seqs in seqs_by_class.items():
        for arr in seqs:
            X.append(arr)
            y.append(lbl)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)

X, y = cache_sequences(BASE_PATH)
print('Data loaded:', X.shape, y.shape, 'Labels:', np.bincount(y))

# ---- SPLIT ----
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---- MODEL ----
def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet', pooling='avg')
    base.trainable = False
    inp = keras.layers.Input(shape=(SEQ_LEN, *IMG_SIZE, 3))
    x = keras.layers.TimeDistributed(base)(inp)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out)

model = build_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint('best_pain_model.keras', monitor='val_accuracy', save_best_only=True)
]

# ---- TRAIN ----
print("\nFAST TRAINING (caching in RAM)...")
model.fit(
    X_train / 255.0,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val / 255.0, y_val),
    callbacks=callbacks,
    verbose=2,
    shuffle=True
)

print("\nAll done! Model and logs saved.")
