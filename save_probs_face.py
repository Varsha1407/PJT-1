import numpy as np
import tensorflow as tf
from collections import Counter

# Load your face test data
X_test_face = np.load('X_test_face.npy')
y_test_face = np.load('y_test_face.npy')   # these are in 0–4

print("Original label counts:", Counter(y_test_face))

# Collapse labels to [0,1,2] as with posture
def collapse_pain_levels_face(y):
    collapsed = []
    for val in y:
        if val == 0:
            collapsed.append(0)
        elif val in [1, 3]:
            collapsed.append(1)
        elif val in [4]:  # if you had 5-7, add here
            collapsed.append(2)
    return np.array(collapsed)

y_test_face_collapsed = collapse_pain_levels_face(y_test_face)
print("Collapsed label counts:", Counter(y_test_face_collapsed))

# Load trained face model
face_model = tf.keras.models.load_model('best_pain_model.keras')

# Predict probabilities
probs_face = face_model.predict(X_test_face / 255.0)        # shape (N, 5)

# Collapse probs_face to shape (N, 3)
probs_face_collapsed = np.zeros((probs_face.shape[0], 3))
probs_face_collapsed[:, 0] = probs_face[:, 0]               # No pain
probs_face_collapsed[:, 1] = probs_face[:, 1] + probs_face[:, 3]  # Low pain
probs_face_collapsed[:, 2] = probs_face[:, 4]               # Mod/high pain (Add classes 4,5,6,7 if present!)

# Save output for fusion
np.save('probs_face.npy', probs_face_collapsed)
np.save('y_test_face_collapsed.npy', y_test_face_collapsed)
print("Saved: probs_face.npy and y_test_face_collapsed.npy")
