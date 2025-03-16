# InsightFace-FaceRecognition

This project implements a real-time Face Recognition System using [InsightFace](https://github.com/deepinsight/insightface). The system efficiently detects and recognizes faces using GPU acceleration, FAISS for embedding lookup, and adaptive thresholding for improved accuracy.

## Features
- **Real-time Face Detection & Recognition** using InsightFace
- **FAISS** for fast embedding lookup
- **Face Tracking** (SORT/DeepSORT) to reduce redundant face processing
- **Multi-threading** for parallel frame processing
- **Adaptive Thresholding** for improved recognition accuracy
- **CLAHE & Auto Brightness Correction** to handle low-light conditions

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/nikh-iam/InsightFace-FaceRecognition.git
cd InsightFace-FaceRecognition
```
### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```sh
pip install -r requirements.txt
```
### 3. Install InsightFace Model
Download and place the `buffalo_s` model inside `~/.insightface/models/` or configure it accordingly.

## Usage
### Run the Face Recognition System
```sh
python main.py
```
Press `q` to exit the application.

### Extract Face Embeddings
```sh
python extract_embeddings.py
```
This script extracts face embeddings from images in the `faces_db/` directory and stores them in `face_embeddings/`.

## Project Structure
```
├── face_recognition.py       # Real-time face recognition script
├── extract_embeddings.py     # Script to extract and store face embeddings
├── face_embeddings/          # Directory for stored face embeddings
├── faces_db/                 # Directory containing face images
├── requirements.txt          # List of dependencies
├── README.md                 # Documentation
```

## How It Works
1. **Face Detection**: Detects faces in video frames using InsightFace.
2. **Face Recognition**: Matches detected faces with stored embeddings.
3. **Face Tracking**: SORT/DeepSORT reduces redundant processing.
4. **Performance Optimization**:
   - Multi-threading for real-time processing
   - FAISS for fast embedding searches
   - TensorRT for GPU acceleration

## Future Improvements
- Support for **multiple cameras**
- **Web-based interface** for monitoring
- **Automatic dataset updates** with real-time learning

## License
This project is licensed under the MIT License.
