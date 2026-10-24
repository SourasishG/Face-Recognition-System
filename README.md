# Face Recognition System using Deep CNN Embeddings & Webcam

An end-to-end Face Recognition System built with Python, OpenCV, PyTorch, and FaceNet. The system captures face images from a live webcam, crops and normalizes them, extracts 512-dimensional facial embeddings using a pre-trained Inception-ResNet CNN, and identifies registered individuals in real time while tagging unregistered people as **"Unknown"**.

---

## Project Tasks & Structure

This project fulfills all 10 core computer vision and machine learning requirements:

| Step / Script | Corresponding Tasks | Description |
| :--- | :--- | :--- |
| `01_collect_data.py` | **Tasks 1 & 2** | Captures 30 facial images via webcam and organizes them into identity-specific folders under `dataset/`. |
| `02_preprocess_and_split.py` | **Tasks 3, 4, 5** | Detects and crops faces, resizes to 160×160, normalizes pixel values to `[-1, 1]`, and splits data into 80% train and 20% test. |
| `03_train_model.py` | **Tasks 6 & 7** | Extracts 512-D deep feature embeddings using pre-trained FaceNet and builds centroid models in `models/face_model.pkl`. |
| `04_evaluate_model.py` | **Tasks 8 & 9** | Tests model against known and unknown faces, calculating Accuracy, Precision, Recall, F1-Score, and saving a Confusion Matrix plot. |
| `05_live_recognition.py` | **Task 10** | Live webcam inference: draws green boxes for registered persons and red boxes for unknown visitors. |

---

## Directory Structure

```text
Face Recognition System/
│
├── dataset/                        # Raw face images captured per user
│   ├── person_1/
│   └── person_2/
│
├── dataset_split/                  # Preprocessed 160x160 images
│   ├── train/                      # 80% training set
│   └── test/                       # 20% testing set
│
├── models/
│   ├── face_model.pkl              # Saved embeddings & registered centroids
│   └── confusion_matrix.png        # Evaluation plot
│
├── 01_collect_data.py              # Task 1 & 2: Dataset collection
├── 02_preprocess_and_split.py      # Task 3, 4, 5: Preprocessing & split
├── 03_train_model.py               # Task 6 & 7: Feature extraction & training
├── 04_evaluate_model.py            # Task 8 & 9: Evaluation & metrics
├── 05_live_recognition.py          # Task 10: Real-time webcam identification
│
├── requirements.txt                # Project dependencies
├── .gitignore                      # Keeps personal photos and cache out of Git
└── README.md                       # Documentation
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/face-recognition-system.git
cd face-recognition-system
```

### 2. Create and activate a virtual environment
```bash
# On Windows PowerShell:
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Project (Step-by-Step)

### Step 1: Register Yourself & Other Users
Run the data collection script:
```bash
python 01_collect_data.py
```
* Enter your name when prompted (e.g. `sourasish`).
* Look at the webcam and move your head slightly while it captures 30 photos.
* Repeat this step to register any additional users (friends/family).

### Step 2: Preprocess and Split the Dataset
```bash
python 02_preprocess_and_split.py
```
* Resizes all images to 160×160 pixels.
* Automatically splits into 80% training and 20% testing sets.

### Step 3: Extract CNN Features & Train
```bash
python 03_train_model.py
```
* Passes face images through pre-trained FaceNet (`InceptionResnetV1`).
* Computes L2-normalized 512-D embeddings and identity centroids.
* Saves database to `models/face_model.pkl`.

### Step 4: Evaluate Model Performance
```bash
python 04_evaluate_model.py
```
* Evaluates accuracy on the test set.
* Prints classification report (Precision, Recall, F1-Score).
* Generates and saves `models/confusion_matrix.png`.

### Step 5: Test Real-Time Webcam Recognition
```bash
python 05_live_recognition.py
```
* Opens the webcam stream.
* Displays a **Green Box** with your name if recognized.
* Displays a **Red Box** with `"Unknown"` if someone unregistered is in front of the camera.
* Press `q` to quit.

---

## Key Concepts Explained

### 1. Face Detection vs. Face Recognition
* **Face Detection** (OpenCV Haar Cascade): Answers *"Where is the face in the image?"*
* **Face Recognition** (FaceNet CNN): Answers *"Whose face is this?"*

### 2. Deep Facial Embeddings
Instead of raw pixels, FaceNet maps faces into a 512-dimensional vector space where:
* Embeddings of the same person have a high **Cosine Similarity** ($\ge 0.65$).
* Embeddings of different people have low similarity.

### 3. Rejecting Unknown Persons
If the maximum cosine similarity to any registered user is less than the threshold ($0.65$), the system marks the face as `"Unknown"` rather than guessing a wrong person.
