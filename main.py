import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import threading
from collections import deque
import os
import pickle
import json

class FaceRecognitionSystem:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_s',
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_faces = {}
        self.frame_queue = deque(maxlen=2)
        self.result_queue = deque(maxlen=2)
        self.processing = False
        self.skip_frames = 2  
        self.frame_count = 0
        self.last_faces = []
        
        self.prev_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        self.thread = None
        self.running = False
        
        self.data_dir = "face_embeddings"
        self.load_embeddings()

    def start_processing(self):
        self.running = True
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.daemon = True
        self.thread.start()

    def stop_processing(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def process_frames(self):
        while self.running:
            if self.frame_queue and not self.processing:
                self.processing = True
                frame = self.frame_queue.popleft()
                
                scale_factor = 0.5
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                
                faces = self.app.get(small_frame)
                
                for face in faces:
                    face.bbox = face.bbox / scale_factor
                    face.kps = face.kps / scale_factor
                
                self.result_queue.append(faces)
                self.processing = False
            else:
                time.sleep(0.01)

    def recognize_face(self, embedding):
        if not self.known_faces:
            return "Unknown", 0
        
        best_match = None
        best_score = -1  # Cosine similarity ranges from -1 to 1

        for name, ref_embedding in self.known_faces.items():
            similarity = np.dot(embedding, ref_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)
            )

            if similarity > best_score:
                best_score = similarity
                best_match = name

        confidence = (best_score + 1) / 2 * 100 

        return best_match if best_score > 0.5 else "Unknown", confidence

    def load_embeddings(self):
        if not os.path.exists(self.data_dir):
            return
        
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                names = metadata.get("names", [])
        else:
            pkl_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
            names = [os.path.splitext(f)[0] for f in pkl_files]
        
        loaded_count = 0
        for name in names:
            safe_name = ''.join(c for c in name if c.isalnum() or c in [' ', '_']).rstrip()
            safe_name = safe_name.replace(' ', '_')
            
            embedding_path = os.path.join(self.data_dir, f"{safe_name}.pkl")
            if os.path.exists(embedding_path):
                try:
                    with open(embedding_path, "rb") as f:
                        embedding = pickle.load(f)
                        self.known_faces[name] = embedding
                        loaded_count += 1
                except Exception as e:
                    print(f"Error loading embedding for {name}: {e}")
        
        if loaded_count > 0:
            print(f"Loaded {loaded_count} face embeddings from {self.data_dir}")

    def is_low_light(self, frame, threshold=70):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = gray[:, :, 0]
        mean_brightness = np.mean(y_channel)
        return mean_brightness < threshold, mean_brightness

    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.start_processing()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                frame = cv2.flip(frame, 1)
                
                self.fps_counter += 1
                current_time = time.time()
                elapsed = current_time - self.prev_time
                if elapsed > 1.0:
                    self.fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.prev_time = current_time
                
                self.frame_count += 1
                if self.frame_count % self.skip_frames == 0:
                    if len(self.frame_queue) < self.frame_queue.maxlen:
                        self.frame_queue.append(frame.copy())
                
                display_frame = frame.copy()
                faces = self.last_faces
                if self.result_queue:
                    faces = self.result_queue.popleft()
                    self.last_faces = faces

                is_dark, brightness = self.is_low_light(frame)

                if is_dark:
                    cv2.putText(display_frame, "Warning: Low Light Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                for face in faces:
                    bbox = face.bbox.astype(int)
                    landmarks = face.kps.astype(int)
                    
                    # Draw landmarks
                    # for landmark in landmarks:
                    #     cv2.circle(display_frame, (landmark[0], landmark[1]), 1, (255, 255, 255), -1)

                    corner_length = 15  
                    color = (255, 255, 255)
                    thickness = 1 

                    # Top-left corner
                    cv2.line(display_frame, (bbox[0], bbox[1]), (bbox[0] + corner_length, bbox[1]), color, thickness)
                    cv2.line(display_frame, (bbox[0], bbox[1]), (bbox[0], bbox[1] + corner_length), color, thickness)

                    # Top-right corner
                    cv2.line(display_frame, (bbox[2], bbox[1]), (bbox[2] - corner_length, bbox[1]), color, thickness)
                    cv2.line(display_frame, (bbox[2], bbox[1]), (bbox[2], bbox[1] + corner_length), color, thickness)

                    # Bottom-left corner
                    cv2.line(display_frame, (bbox[0], bbox[3]), (bbox[0] + corner_length, bbox[3]), color, thickness)
                    cv2.line(display_frame, (bbox[0], bbox[3]), (bbox[0], bbox[3] - corner_length), color, thickness)

                    # Bottom-right corner
                    cv2.line(display_frame, (bbox[2], bbox[3]), (bbox[2] - corner_length, bbox[3]), color, thickness)
                    cv2.line(display_frame, (bbox[2], bbox[3]), (bbox[2], bbox[3] - corner_length), color, thickness)

                    # Recognize face
                    recognized_name, confidence = self.recognize_face(face.embedding)
                    label = f"{recognized_name} ({confidence:.1f}%)" if recognized_name != "Unknown" else "Unknown"
                    
                    # Display label above the detection
                    cv2.putText(display_frame, label.capitalize(), (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (display_frame.shape[1] - 120, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                cv2.imshow('Face Recognition', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            self.stop_processing()
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
