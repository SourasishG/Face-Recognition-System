import os
import cv2
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from facenet_pytorch import InceptionResnetV1

# Task 8 & 9: Test Model with Known/Unknown Persons and Calculate Performance Metrics
device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load trained centroids and model
with open("models/face_model.pkl", "rb") as f:
    saved_data = pickle.load(f)

database = saved_data["centroids"]
threshold = 0.65  # Cosine similarity threshold for unknown rejection

test_dir = os.path.join("dataset_split", "test")
y_true = []
y_pred = []

print(f"[INFO] Evaluating model with similarity threshold = {threshold}...")

for person in os.listdir(test_dir):
    person_dir = os.path.join(test_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        image = cv2.imread(os.path.join(person_dir, filename))
        if image is None:
            continue

        # Preprocess image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0
        tensor = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)

        # Compare with registered centroids
        best_match = "unknown"
        best_score = -1.0

        for name, centroid in database.items():
            # Cosine similarity is dot product of normalized vectors
            similarity = float(np.dot(emb, centroid))
            if similarity > best_score:
                best_score = similarity
                best_match = name

        # Decision threshold for unknown faces
        if best_score >= threshold:
            prediction = best_match
        else:
            prediction = "unknown"

        y_true.append(person)
        y_pred.append(prediction)

# Calculate Metrics
accuracy = accuracy_score(y_true, y_pred)
print("\n" + "="*50)
print(f"Overall Recognition Accuracy: {accuracy * 100:.2f}%")
print("="*50)
print("\nClassification Report (Precision, Recall, F1-Score):")
print(classification_report(y_true, y_pred, zero_division=0))

# Confusion Matrix
labels = sorted(list(set(y_true + y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("Confusion Matrix:")
print(cm)

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap="Blues")
fig.colorbar(cax)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45)
ax.set_yticklabels(labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Face Recognition Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("\n[SUCCESS] Confusion matrix saved as 'models/confusion_matrix.png'.")
