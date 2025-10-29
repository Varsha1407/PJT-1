import tensorflow as tf
from tensorflow import keras
import pandas as pd

print("\n=== Checking Face Model Summary ===")
face_model = keras.models.load_model("best_pain_model.keras", compile=False)
face_model.summary()

# Get embedding output shape safely
try:
    embedding_shape = face_model.layers[-2].output.shape
    print("\nEmbedding Layer:", face_model.layers[-2].name)
    print("Embedding Output Shape:", embedding_shape)
except Exception as e:
    print("\nCould not get output shape directly:", e)
    dummy_input = tf.random.normal((1, 12, 64, 64, 3))
    intermediate_model = keras.Model(inputs=face_model.input, outputs=face_model.layers[-2].output)
    dummy_output = intermediate_model(dummy_input)
    print("Embedding shape (via test input):", dummy_output.shape)

print("\n=== Checking Posture Pain Levels ===")
df = pd.read_csv("extracted_features.csv")

if "pain_level" in df.columns:
    unique_labels = sorted(df["pain_level"].dropna().unique())
    print(f"Unique pain levels found in CSV: {unique_labels}")
    print(f"Total classes detected: {len(unique_labels)}")
else:
    print("Column 'pain_level' not found in the CSV. Please verify the column name.")
