import os
import cv2
import pickle
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

# Task 6 & 7: Extract Facial Features (CNN) and Train Recognition Model
device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

print(f"[INFO] Using device: {device}")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

train_dir = os.path.join("dataset_split", "train")
embeddings_list = []
labels_list = []

print("[INFO] Extracting CNN embeddings for training set...")

for person in os.listdir(train_dir):
    person_dir = os.path.join(train_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(person_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert BGR to RGB and normalize to [-1, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0

        # Convert to PyTorch Tensor: Shape (1, 3, 160, 160)
        tensor = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(tensor).cpu().numpy()[0]
            # L2 normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)

        embeddings_list.append(embedding)
        labels_list.append(person)

# Calculate average centroid embedding for each registered person
user_database = {}
for emb, label in zip(embeddings_list, labels_list):
    if label not in user_database:
        user_database[label] = []
    user_database[label].append(emb)

centroid_database = {}
for label in user_database:
    centroid_database[label] = np.mean(user_database[label], axis=0)
    # L2 normalize the centroid vector
    centroid_database[label] = centroid_database[label] / np.linalg.norm(centroid_database[label])

# Save model and embeddings
os.makedirs("models", exist_ok=True)
save_data = {
    "centroids": centroid_database,
    "all_embeddings": np.array(embeddings_list),
    "all_labels": labels_list
}

with open("models/face_model.pkl", "wb") as f:
    pickle.dump(save_data, f)

print(f"[SUCCESS] Trained model on {len(labels_list)} faces across {len(centroid_database)} identities.")
print("[SUCCESS] Saved face model to 'models/face_model.pkl'.")
