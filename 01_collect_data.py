import cv2
import os

# Task 1 & 2: Collect face dataset and organize by person identity
person_name = input("Enter the person's name: ").strip().lower()
save_folder = os.path.join("dataset", person_name)
os.makedirs(save_folder, exist_ok=True)

# Load OpenCV's built-in Haar Cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
count = 0
total_samples = 30

print("[INFO] Starting webcam. Look at the camera and move your head slightly.")
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
        # Crop the detected face
        face_crop = frame[y:y+h, x:x+w]

        # Save the cropped face image
        file_path = os.path.join(save_folder, f"face_{count}.jpg")
        cv2.imwrite(file_path, face_crop)
        count += 1

        # Draw green bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break

    # Display progress on screen
    text = f"Samples: {count}/{total_samples}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Registering Face", frame)

    if count >= total_samples:
        print(f"[SUCCESS] Collected {total_samples} images for {person_name}.")
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("[INFO] Process interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()

