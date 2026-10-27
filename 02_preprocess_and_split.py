import os
import cv2
import random

# Task 3, 4, 5: Detect/Crop, Resize, Normalize, and Split Dataset
dataset_dir = "dataset"
output_train = os.path.join("dataset_split", "train")
output_test = os.path.join("dataset_split", "test")

# Standard size for FaceNet CNN
target_size = (160, 160)
train_ratio = 0.8  # 80% train, 20% test

persons = os.listdir(dataset_dir)

for person in persons:
    person_path = os.path.join(dataset_dir, person)
    
    # Check if it is a directory and not a hidden file
    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)
    # Filter only image files
    valid_images = []
    for img in images:
        if img.endswith(".jpg"):
            valid_images.append(img)
        elif img.endswith(".png"):
            valid_images.append(img)

    random.shuffle(valid_images)
    split_index = int(len(valid_images) * train_ratio)
    train_files = valid_images[:split_index]
    test_files = valid_images[split_index:]

    # Create output folders for this person
    train_save_dir = os.path.join(output_train, person)
    test_save_dir = os.path.join(output_test, person)
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    # Process and save images
    for group_files, target_dir in [(train_files, train_save_dir), (test_files, test_save_dir)]:
        for filename in group_files:
            file_path = os.path.join(person_path, filename)
            image = cv2.imread(file_path)
            
            if image is None:
                continue

            # Task 4: Resize to 160x160
            resized_face = cv2.resize(image, target_size)

            # Save the processed face image
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, resized_face)

    print(f"[INFO] {person}: {len(train_files)} train images, {len(test_files)} test images.")

print("[SUCCESS] Dataset preprocessing and train/test split is completed.")

