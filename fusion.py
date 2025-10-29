import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
Model = keras.Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


# ===== 1. Load and preprocess posture data =====
df_posture = pd.read_csv('extracted_features.csv')

non_numeric_cols = df_posture.select_dtypes(include=['object']).columns.tolist()
print("Dropping non-numeric columns:", non_numeric_cols)
if 'pain_level' in non_numeric_cols:
    non_numeric_cols.remove('pain_level')
df_posture_numeric = df_posture.drop(columns=non_numeric_cols)

X_posture = df_posture_numeric.drop(columns=['pain_level']).values
y_posture = df_posture_numeric['pain_level'].astype(int).values

valid_classes = set(range(5))
mask = np.isin(y_posture, list(valid_classes))
X_posture = X_posture[mask]
y_posture = y_posture[mask]

print("Initial label counts:", Counter(y_posture))

counts = Counter(y_posture)
filtered_classes = [cls for cls, count in counts.items() if count >= 2]
final_mask = np.isin(y_posture, filtered_classes)
X_posture = X_posture[final_mask]
y_posture = y_posture[final_mask]

print("Filtered label counts:", Counter(y_posture))

classes = np.unique(y_posture)
class_weights = compute_class_weight('balanced', classes=classes, y=y_posture)
class_weight_dict = {int(k): v for k, v in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

scaler_posture = StandardScaler()
X_posture_scaled = scaler_posture.fit_transform(X_posture)
SEQ_LEN = 12
X_posture_temporal = np.repeat(X_posture_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)

X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
    X_posture_temporal, y_posture,
    test_size=0.2,
    stratify=y_posture,
    random_state=42
)

print(f"Posture temporal shape: {X_train_pos.shape}")


# ===== 2. Build and train posture model =====
def build_posture_temporal_model(input_shape, num_classes=5):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(128)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

posture_model = build_posture_temporal_model((SEQ_LEN, X_posture_scaled.shape[1]), 5)
posture_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
posture_model.summary()

print("Training Posture Model with class weights...")
posture_model.fit(X_train_pos, y_train_pos,
                  epochs=25,
                  batch_size=16,
                  validation_split=0.1,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                  class_weight=class_weight_dict,
                  verbose=2)

# ===== 3. Load face test data and pretrained model =====
X_test_face = np.load('X_test_face.npy')
y_test_face = np.load('y_test_face.npy')
temporal_model = keras.models.load_model('best_pain_model.keras')

# ===== 4. Predict probabilities =====
probs_posture = posture_model.predict(X_test_pos)
probs_face = temporal_model.predict(X_test_face)
y_pred_posture = np.argmax(probs_posture, axis=1)
y_pred_face = np.argmax(probs_face, axis=1)


# ===== 5. Classification reports and confusion matrices =====
print("\nPosture Model Report")
print(classification_report(y_test_pos, y_pred_posture))
cm_posture = confusion_matrix(y_test_pos, y_pred_posture)
sns.heatmap(cm_posture, annot=True, fmt='d', cmap='Blues')
plt.title('Posture Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nFace Model Report")
print(classification_report(y_test_face, y_pred_face))
cm_face = confusion_matrix(y_test_face, y_pred_face)
sns.heatmap(cm_face, annot=True, fmt='d', cmap='Purples')
plt.title('Face Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ===== 6. Meta Fusion Detection Rates =====
pain_classes = [1, 2, 3, 4]
high_pain_classes = [3, 4]

# Lower detection threshold to catch more positives
threshold_any_pain = 0.3
threshold_high_pain = 0.3

pain_posture = np.sum(probs_posture[:, pain_classes], axis=1)
pain_face = np.sum(probs_face[:, pain_classes], axis=1)
posture_pain_detected = pain_posture > threshold_any_pain
face_pain_detected = pain_face > threshold_any_pain
meta_pain_detected = np.concatenate([posture_pain_detected, face_pain_detected]).astype(int)
meta_pain_rate = np.mean(meta_pain_detected)

high_pain_posture = np.sum(probs_posture[:, high_pain_classes], axis=1)
high_pain_face = np.sum(probs_face[:, high_pain_classes], axis=1)
posture_high_detected = high_pain_posture > threshold_high_pain
face_high_detected = high_pain_face > threshold_high_pain
meta_high_detected = np.concatenate([posture_high_detected, face_high_detected]).astype(int)
meta_high_rate = np.mean(meta_high_detected)

print("\nDetection Rates Summary:")
print(f"Posture Any Pain Detection Rate: {np.mean(posture_pain_detected):.2f}")
print(f"Face Any Pain Detection Rate: {np.mean(face_pain_detected):.2f}")
print(f"Meta Fusion Any Pain Detection Rate: {meta_pain_rate:.2f}")
print(f"Posture High Pain Detection Rate: {np.mean(posture_high_detected):.2f}")
print(f"Face High Pain Detection Rate: {np.mean(face_high_detected):.2f}")
print(f"Meta Fusion High Pain Detection Rate: {meta_high_rate:.2f}")

# ===== 7. Plot results =====
labels = ['Posture', 'Face', 'Meta Fusion']
pain_rates = [np.mean(posture_pain_detected), np.mean(face_pain_detected), meta_pain_rate]
high_pain_rates = [np.mean(posture_high_detected), np.mean(face_high_detected), meta_high_rate]

plt.bar(labels, pain_rates, color=['blue', 'orange', 'purple'])
plt.title("Any Pain Detection Rate by Modality")
plt.ylabel("Detection Rate")
plt.ylim(0, 1)
plt.show()

plt.bar(labels, high_pain_rates, color=['blue', 'orange', 'purple'])
plt.title("High Pain Detection Rate by Modality")
plt.ylabel("Detection Rate")
plt.ylim(0, 1)
plt.show()

# ===== 8. Save to CSV =====
fusion_results = pd.DataFrame({
    'Modality': labels,
    'Any Pain Detection Rate': pain_rates,
    'High Pain Detection Rate': high_pain_rates
})
fusion_results.to_csv('meta_fusion_results.csv', index=False)
print(fusion_results)
