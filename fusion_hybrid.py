import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd

# ===== 1. Load Logits & Labels for Both Modalities =====
probs_face = np.load('probs_face.npy').astype(np.float32)          # Face model probs (N, 3)
probs_posture = np.load('probs_posture.npy').astype(np.float32)    # Posture model probs (M, 3)
y_test_face = np.load('y_test_face_collapsed.npy')
y_test_posture = np.load('y_test_posture.npy')
NUM_CLASSES = 3  # Because of collapsed classes

# ===== 2. Create Pseudo-Pairs By Label =====
def make_pseudo_pairs(probs_face, y_face, probs_posture, y_posture, max_pairs_per_class=400):
    pairs = []
    for c in range(NUM_CLASSES):
        idx_face = np.where(y_face == c)[0]
        idx_posture = np.where(y_posture == c)[0]
        n = min(len(idx_face), len(idx_posture), max_pairs_per_class)
        if n == 0:
            continue
        for i in range(n):
            pairs.append((probs_face[idx_face[i]], probs_posture[idx_posture[i]], c))
    random.shuffle(pairs)
    return pairs

pairs = make_pseudo_pairs(probs_face, y_test_face, probs_posture, y_test_posture)
face_logits = np.array([a for a, b, lbl in pairs]).astype(np.float32)
posture_logits = np.array([b for a, b, lbl in pairs]).astype(np.float32)
labels = np.array([lbl for a, b, lbl in pairs])

print(f"Number of pseudo-pairs: {len(labels)}")
print(f"Class distribution in pairs: {np.bincount(labels)}")

# ===== Compute class weights =====
classes = np.unique(labels)
class_weights_arr = compute_class_weight('balanced', classes=classes, y=labels)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
print("Fusion class weights:", class_weight_dict)

# ===== 3. Fusion MLP Model =====
inputs = keras.layers.Input(shape=(NUM_CLASSES*2,))  # 6
x = keras.layers.Dense(32, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
fusion_model = keras.Model(inputs, out)
fusion_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===== 4. Hybrid Meta-Fusion Training Loop with Class Weighting =====
def hybrid_loss(y_true, y_pred, face_logits, posture_logits):
    y_pred = tf.cast(y_pred, tf.float32)
    face_logits = tf.cast(face_logits, tf.float32)
    posture_logits = tf.cast(posture_logits, tf.float32)
    ce_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    fusion_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    face_norm = tf.nn.l2_normalize(face_logits, axis=-1)
    posture_norm = tf.nn.l2_normalize(posture_logits, axis=-1)
    contrastive_loss = 1.0 - tf.reduce_mean(tf.reduce_sum(fusion_pred_norm * face_norm, axis=-1))
    contrastive_loss += 1.0 - tf.reduce_mean(tf.reduce_sum(fusion_pred_norm * posture_norm, axis=-1))
    kl_loss = keras.losses.KLDivergence()(tf.nn.softmax(face_logits), y_pred)
    kl_loss += keras.losses.KLDivergence()(tf.nn.softmax(posture_logits), y_pred)
    total_loss = ce_loss + 0.5 * contrastive_loss + 0.5 * kl_loss
    return total_loss

class HybridLossTrainer:
    def __init__(self, model, class_weight):
        self.model = model
        self.class_weight = class_weight
        self.optimizer = keras.optimizers.Adam()

    def train_step(self, X, y, face, posture):
        weights = tf.convert_to_tensor([self.class_weight[int(label)] for label in y], dtype=tf.float32)
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        face = tf.cast(face, tf.float32)
        posture = tf.cast(posture, tf.float32)
        with tf.GradientTape() as tape:
            y_pred = self.model(X, training=True)
            losses = hybrid_loss(y, y_pred, face, posture)
            weighted_loss = tf.reduce_sum(losses * weights) / tf.reduce_sum(weights)
        grads = tape.gradient(weighted_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return weighted_loss.numpy()

X_fusion = np.concatenate([face_logits, posture_logits], axis=-1).astype(np.float32)
y_fusion = labels.astype(np.float32)
trainer = HybridLossTrainer(fusion_model, class_weight_dict)

epochs = 30
batch_size = 32
for epoch in range(epochs):
    idx = np.arange(len(X_fusion))
    np.random.shuffle(idx)
    losses = []
    for i in range(0, len(idx), batch_size):
        batch = idx[i:i+batch_size]
        loss = trainer.train_step(X_fusion[batch], y_fusion[batch], face_logits[batch], posture_logits[batch])
        losses.append(loss)
    print(f"Epoch {epoch+1}: Mean Hybrid Loss = {np.mean(losses):.4f}")

# ===== 5. Metrics, Confusion Matrix, Plots =====
y_pred_fusion = np.argmax(fusion_model.predict(X_fusion.astype(np.float32)), axis=1)
y_true_fusion = labels

acc = accuracy_score(y_true_fusion, y_pred_fusion)
f1 = f1_score(y_true_fusion, y_pred_fusion, average='weighted')
precision = precision_score(y_true_fusion, y_pred_fusion, average='weighted')
recall = recall_score(y_true_fusion, y_pred_fusion, average='weighted')
print(f"\nHybrid Meta-Fusion Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f} Precision: {precision:.2f} Recall: {recall:.2f}\n")

print("Classification Report:\n", classification_report(y_true_fusion, y_pred_fusion))

cm = confusion_matrix(y_true_fusion, y_pred_fusion)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title("Hybrid Meta-Fusion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

pain_classes = [1, 2]
high_pain_classes = [2]

probs_fusion = fusion_model.predict(X_fusion.astype(np.float32))
pain_detected = np.sum(probs_fusion[:, pain_classes], axis=1) > 0.3
high_pain_detected = np.sum(probs_fusion[:, high_pain_classes], axis=1) > 0.3
pain_rate = np.mean(pain_detected)
high_pain_rate = np.mean(high_pain_detected)
print(f"Hybrid Any Pain Detection Rate: {pain_rate:.2f}")
print(f"Hybrid High Pain Detection Rate: {high_pain_rate:.2f}")

plt.bar(['Hybrid Fusion'], [pain_rate], color='purple')
plt.title("Hybrid Any Pain Detection Rate")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.show()

plt.bar(['Hybrid Fusion'], [high_pain_rate], color='orange')
plt.title("Hybrid High Pain Detection Rate")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.show()

results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'Any Pain Detection Rate', 'High Pain Detection Rate'],
    'Value': [acc, f1, precision, recall, pain_rate, high_pain_rate]
})
results_df.to_csv('hybrid_fusion_metrics.csv', index=False)
print("Results saved to hybrid_fusion_metrics.csv")

# ... (your existing code above remains unchanged)

# Save confusion matrix plot
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title("Hybrid Meta-Fusion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# Save pain detection rate bar plots
plt.figure(figsize=(6,4))
plt.bar(['Hybrid Fusion'], [pain_rate], color='purple')
plt.title("Hybrid Any Pain Detection Rate")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.savefig("any_pain_detection_rate.png")
plt.close()

plt.figure(figsize=(6,4))
plt.bar(['Hybrid Fusion'], [high_pain_rate], color='orange')
plt.title("Hybrid High Pain Detection Rate")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.savefig("high_pain_detection_rate.png")
plt.close()

# Save metrics to CSV
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'Any Pain Detection Rate', 'High Pain Detection Rate'],
    'Value': [acc, f1, precision, recall, pain_rate, high_pain_rate]
})
results_df.to_csv('hybrid_fusion_metrics.csv', index=False)

print("Plots saved as PNG files.")
print("Metrics saved to hybrid_fusion_metrics.csv")
