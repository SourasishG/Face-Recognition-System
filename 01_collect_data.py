import cv2
import os

# Task 1 & 2: Collect face dataset and organize by person identity
person_name = input("Enter the person's name: ").strip().lower()
save_folder = os.path.join("dataset", person_name)
os.makedirs(save_folder, exist_ok=True)

# Count existing images so we don't overwrite them
existing_images = os.listdir(save_folder)
image_index = len(existing_images)

# Load OpenCV's built-in Haar Cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
count = 0
samples_to_take = 30
pad = 20  # Padding around face to capture glasses and full head structure

print(f"[INFO] Saving images for {person_name}. Current total: {image_index}")
print("[INFO] Press 'q' to exit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from webcam.")
        break

    # Flip horizontally so it behaves like a natural mirror
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop face with padding
        h_img, w_img = frame.shape[:2]
        y1 = max(0, y - pad)
        y2 = min(h_img, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(w_img, x + w + pad)
        face_crop = frame[y1:y2, x1:x2]

        # Save image
        file_path = os.path.join(save_folder, f"face_{image_index}.jpg")
        cv2.imwrite(file_path, face_crop)
        image_index += 1
        count += 1

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        break

    # Display progress
    text = f"New Samples: {count}/{samples_to_take} (Total: {image_index})"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Registering Face", frame)

    if count >= samples_to_take:
        print(f"[SUCCESS] Finished capturing {samples_to_take} images for {person_name}.")
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("[INFO] Process interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()
