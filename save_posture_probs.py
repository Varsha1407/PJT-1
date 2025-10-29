import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ===== 1. Load and preprocess posture data =====
df_posture = pd.read_csv('extracted_features.csv')

# Remove all non-numeric columns except 'pain_level'
non_numeric_cols = df_posture.select_dtypes(include=['object']).columns.tolist()
if 'pain_level' in non_numeric_cols:
    non_numeric_cols.remove('pain_level')
df_posture_numeric = df_posture.drop(columns=non_numeric_cols)

X_posture = df_posture_numeric.drop(columns=['pain_level']).values
y_posture = df_posture_numeric['pain_level'].astype(int).values

print("Original label counts:", Counter(y_posture))

# ===== Collapse and filter for 3-class problem (ensuring matched arrays) =====
features_filtered = []
labels_collapsed = []
for feat, orig_label in zip(X_posture, y_posture):
    if orig_label == 0:
        labels_collapsed.append(0)
        features_filtered.append(feat)
    elif orig_label in [1, 3]:
        labels_collapsed.append(1)
        features_filtered.append(feat)
    elif orig_label in [4, 5, 6, 7]:
        labels_collapsed.append(2)
        features_filtered.append(feat)
    # All other labels are ignored

X_posture_final = np.array(features_filtered)
y_posture_final = np.array(labels_collapsed)
print("Final Shapes (should match):", X_posture_final.shape, y_posture_final.shape)
print("Collapsed label counts:", Counter(y_posture_final))

# ===== Standardize features =====
scaler_posture = StandardScaler()
X_posture_scaled = scaler_posture.fit_transform(X_posture_final)
SEQ_LEN = 12
X_posture_temporal = np.repeat(X_posture_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)

# ===== Train/test split =====
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
    X_posture_temporal, y_posture_final,
    test_size=0.2,
    stratify=y_posture_final,
    random_state=42
)
print("Train label counts:", Counter(y_train_pos))
print("Test label counts:", Counter(y_test_pos))

# ===== Compute class weights =====
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_pos), y=y_train_pos)
class_weight_dict = {int(k): v for k, v in zip(np.unique(y_train_pos), class_weights)}
print("Class weights:", class_weight_dict)

# ===== Build and train model =====
def build_posture_temporal_model(input_shape, num_classes):
    inp = keras.layers.Input(shape=input_shape)
    x = keras.layers.LSTM(64, return_sequences=True)(inp)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inp, out)

num_classes = 3

posture_model = build_posture_temporal_model((SEQ_LEN, X_posture_scaled.shape[1]), num_classes)
posture_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

posture_model.summary()

history = posture_model.fit(
    X_train_pos, y_train_pos,
    epochs=75,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weight_dict,
    shuffle=True,
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)
    ]
)

# ===== Evaluate and plot confusion matrix =====
probs_test = posture_model.predict(X_test_pos)
y_pred_test = np.argmax(probs_test, axis=1)

print("\nClassification Report:\n", classification_report(y_test_pos, y_pred_test, digits=4))
cm = confusion_matrix(y_test_pos, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Posture Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ===== Save for fusion =====
np.save('probs_posture.npy', probs_test)
np.save('y_test_posture.npy', y_test_pos)
print("Saved: probs_posture.npy and y_test_posture.npy")
