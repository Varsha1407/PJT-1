import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

NUM_CLASSES = 3
REPEATS = 30  # More repeats for more stable averaged metrics

def focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        ce = - y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * ce
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

model = keras.models.load_model('fusion_mlp_best_model.h5', custom_objects={'focal_loss_fixed': focal_loss()})

probs_face = np.load('probs_face.npy').astype(np.float32)
probs_posture = np.load('probs_posture.npy').astype(np.float32)
y_test_face = np.load('y_test_face_collapsed.npy')
y_test_posture = np.load('y_test_posture.npy')

def make_balanced_pseudo_pairs(probs_face, y_face, probs_posture, y_posture):
    pairs = []
    class_counts_face = [len(np.where(y_face == c)[0]) for c in range(NUM_CLASSES)]
    class_counts_posture = [len(np.where(y_posture == c)[0]) for c in range(NUM_CLASSES)]
    min_count = min([count for count in class_counts_face + class_counts_posture if count > 0])
    for c in range(NUM_CLASSES):
        idx_face = np.where(y_face == c)[0]
        idx_posture = np.where(y_posture == c)[0]
        if len(idx_face) == 0 or len(idx_posture) == 0:
            continue
        np.random.shuffle(idx_face)
        np.random.shuffle(idx_posture)
        for i in range(min_count):
            pairs.append((probs_face[idx_face[i]], probs_posture[idx_posture[i]], c))
    random.shuffle(pairs)
    return pairs

metrics_list = []

for seed in range(REPEATS):
    np.random.seed(seed)
    random.seed(seed)
    pairs = make_balanced_pseudo_pairs(probs_face, y_test_face, probs_posture, y_test_posture)
    face_logits = np.array([a for a, b, lbl in pairs])
    posture_logits = np.array([b for a, b, lbl in pairs])
    labels = np.array([lbl for a, b, lbl in pairs])

    X_test = np.concatenate([face_logits, posture_logits], axis=-1)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average='weighted', zero_division=0)
    precision = precision_score(labels, y_pred, average='weighted', zero_division=0)
    recall = recall_score(labels, y_pred, average='weighted', zero_division=0)

    metrics_list.append([acc, f1, precision, recall])

metrics_array = np.array(metrics_list)
metrics_mean = metrics_array.mean(axis=0)
metrics_std = metrics_array.std(axis=0)

df_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
    'Mean': metrics_mean,
    'Std': metrics_std
})

df_metrics.to_csv('test_balanced_metrics_summary.csv', index=False)
print("Balanced repeated test evaluation complete:")
print(df_metrics)

# Save confusion matrix and classification report for final seed
pairs_final = make_balanced_pseudo_pairs(probs_face, y_test_face, probs_posture, y_test_posture)
face_logits = np.array([a for a, b, lbl in pairs_final])
posture_logits = np.array([b for a, b, lbl in pairs_final])
labels = np.array([lbl for a, b, lbl in pairs_final])

X_test_final = np.concatenate([face_logits, posture_logits], axis=-1)
y_pred_probs_final = model.predict(X_test_final)
y_pred_final = np.argmax(y_pred_probs_final, axis=1)

report = classification_report(labels, y_pred_final, zero_division=0)
report_df = pd.DataFrame(classification_report(labels, y_pred_final, output_dict=True, zero_division=0)).transpose()
report_df.to_csv('test_classification_report.csv', index=True)

cm = confusion_matrix(labels, y_pred_final)
plt.figure(figsize=(8,6))
sns.heatmap(cm / cm.sum(axis=1)[:, None], annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['No Pain', 'Low Pain', 'High Pain'],
            yticklabels=['No Pain', 'Low Pain', 'High Pain'])
plt.title('Test Confusion Matrix (%)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png')
plt.close()

print("Confusion matrix and classification report saved.")
