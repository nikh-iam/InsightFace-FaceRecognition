import cv2
import numpy as np
import os
import pickle
import json
from insightface.app import FaceAnalysis

# Paths
TEST_DATASET_PATH = "test_images"
EMBEDDINGS_PATH = "face_embeddings"

# Load Face Analysis Model
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load Known Face Embeddings
def load_embeddings():
    known_faces = {}
    metadata_path = os.path.join(EMBEDDINGS_PATH, "metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            names = metadata.get("names", [])
    else:
        pkl_files = [f for f in os.listdir(EMBEDDINGS_PATH) if f.endswith('.pkl')]
        names = [os.path.splitext(f)[0] for f in pkl_files]
    
    for name in names:
        safe_name = name.replace(' ', '_')
        embedding_path = os.path.join(EMBEDDINGS_PATH, f"{safe_name}.pkl")
        if os.path.exists(embedding_path):
            with open(embedding_path, "rb") as f:
                embedding = pickle.load(f)
                known_faces[name] = embedding
    return known_faces

# Recognize Face
def recognize_face(embedding, known_faces):
    best_match = "Unknown"
    best_score = -1  

    for name, ref_embedding in known_faces.items():
        similarity = np.dot(embedding, ref_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)
        )

        if similarity > best_score:
            best_score = similarity
            best_match = name

    confidence = (best_score + 1) / 2 * 100  

    return best_match if best_score > 0.5 else "Unknown", confidence

# Evaluate Accuracy
# Evaluate Accuracy
def evaluate_accuracy():
    known_faces = load_embeddings()
    
    if not known_faces:
        print("No known faces found. Please add face embeddings first.")
        return
    
    correct = 0
    total = 0
    
    for person_name in os.listdir(TEST_DATASET_PATH):
        person_path = os.path.join(TEST_DATASET_PATH, person_name)
        
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_app.get(image)

            if len(faces) == 1:
                predicted_name, confidence = recognize_face(faces[0].embedding, known_faces)

                # ✅ Fix: Handle unknown faces correctly
                if person_name.lower() == "unknown":
                    if predicted_name == "Unknown":
                        correct += 1  # Count this as a correct prediction
                else:
                    if predicted_name == person_name:
                        correct += 1
                
                total += 1
                print(f"Image: {img_name}, Actual: {person_name}, Predicted: {predicted_name}, Confidence: {confidence:.2f}%")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")


if __name__ == "__main__":
    evaluate_accuracy()
