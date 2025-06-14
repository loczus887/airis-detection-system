import cv2
import time
import numpy as np
from scipy.spatial import distance

class FaceDetector:
    def __init__(self, threshold=0.6, time_limit=10):
        self.threshold = threshold
        self.time_limit = time_limit
        self.faces = {}
        self.debug_mode = False
        
        # Import dlib here to avoid global import
        try:
            import dlib
            # Initialize dlib's face detector
            self.detector = dlib.get_frontal_face_detector()
            
            # Loading landmark predictor for more accurate face detection and identification
            try:
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                self.shape_predictor = dlib.shape_predictor(predictor_path)
                
                # Optional - facial recognition model for better identification
                face_rec_path = "dlib_face_recognition_resnet_model_v1.dat"
                self.face_rec_model = dlib.face_recognition_model_v1(face_rec_path)
                
                self.advanced_recognition = True
                print("Advanced face recognition initialized with dlib models")
            except Exception as e:
                print(f"Warning: Advanced face recognition not available: {e}")
                print("Using basic face tracking instead")
                self.advanced_recognition = False
            
            self.use_dlib = True
            print("Face detector initialized with dlib")
        except ImportError:
            print("WARNING: dlib not found, using OpenCV's face detector instead")
            # Fallback to OpenCV's Haar Cascade classifier if dlib is not available
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(self.cascade_path)
            self.use_dlib = False
            self.advanced_recognition = False
    
    def get_face_embedding(self, face_chip):
        """Calculate face embedding for better identification"""
        if self.advanced_recognition:
            try:
                # Get face descriptor (128D vector)
                return np.array(self.face_rec_model.compute_face_descriptor(face_chip))
            except Exception as e:
                print(f"Error getting face embedding: {e}")
        return None
    
    def process(self, frame):
        """Process a frame to detect and track faces"""
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_rects = []
        landmarks_list = []
        embeddings_list = []
        
        if self.use_dlib:
            try:
                # Use dlib for detection
                dlib_faces = self.detector(gray, 0)  # 0 = no upsampling for speed
                
                for face in dlib_faces:
                    face_rects.append((face.left(), face.top(), face.width(), face.height()))
                    
                    # Get landmarks if advanced recognition is enabled
                    if self.advanced_recognition:
                        try:
                            landmarks = self.shape_predictor(gray, face)
                            landmarks_list.append(landmarks)
                            
                            # Create face chip (aligned face) for recognition
                            face_chip = dlib.get_face_chip(frame, landmarks)
                            embedding = self.get_face_embedding(face_chip)
                            embeddings_list.append(embedding)
                        except Exception as e:
                            if self.debug_mode:
                                print(f"Error processing face landmarks: {e}")
                            embeddings_list.append(None)
                            landmarks_list.append(None)
            except Exception as e:
                print(f"Dlib face detection error: {e}")
                face_rects = []
        else:
            # Use OpenCV's Haar Cascade
            try:
                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                face_rects = [(x, y, w, h) for (x, y, w, h) in faces]
            except Exception as e:
                print(f"OpenCV face detection error: {e}")
                face_rects = []
        
        # Track faces and their durations
        current_time = time.time()
        current_faces = {}
        
        for i, (x, y, w, h) in enumerate(face_rects):
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Create a face ID based on position or embedding
            face_id = None
            
            # If advanced recognition is enabled and we have an embedding
            if self.advanced_recognition and i < len(embeddings_list) and embeddings_list[i] is not None:
                # Try to match with existing faces using embedding
                current_embedding = embeddings_list[i]
                min_distance = float("inf")
                closest_face_id = None
                
                for existing_id, face_data in self.faces.items():
                    if 'embedding' in face_data and face_data['embedding'] is not None:
                        dist = distance.euclidean(current_embedding, face_data['embedding'])
                        if dist < self.threshold and dist < min_distance:
                            min_distance = dist
                            closest_face_id = existing_id
                
                if closest_face_id is not None:
                    face_id = closest_face_id
                else:
                    # New face - generate a unique ID
                    face_id = f"face_{len(self.faces) + 1}_{current_time}"
            else:
                # Fallback to position-based tracking if advanced recognition not available
                # Using center position to be more stable than top-left corner
                center_x, center_y = x + w//2, y + h//2
                face_position = (center_x, center_y)
                
                # Try to match with existing faces using position
                min_distance = float("inf")
                closest_face_id = None
                
                for existing_id, face_data in self.faces.items():
                    if 'position' in face_data:
                        existing_pos = face_data['position']
                        # Calculate Euclidean distance between face centers
                        dist = np.sqrt((face_position[0] - existing_pos[0])**2 + 
                                      (face_position[1] - existing_pos[1])**2)
                        
                        # If the face is close enough to an existing face, it's the same face
                        if dist < max(w, h) * 0.5:  # Threshold based on face size
                            if dist < min_distance:
                                min_distance = dist
                                closest_face_id = existing_id
                
                if closest_face_id is not None:
                    face_id = closest_face_id
                else:
                    # New face - generate a unique ID
                    face_id = f"face_{len(self.faces) + 1}_{current_time}"
            
            current_faces[face_id] = True
            
            # If this is a new face, add it to tracking
            if face_id not in self.faces:
                self.faces[face_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'position': (center_x, center_y),
                    'rect': (x, y, w, h),
                    'alert': False,
                    'total_time': 0,
                    'disappeared_time': 0
                }
                
                # Store embedding for advanced recognition
                if self.advanced_recognition and i < len(embeddings_list) and embeddings_list[i] is not None:
                    self.faces[face_id]['embedding'] = embeddings_list[i]
                
                if self.debug_mode:
                    print(f"New face {face_id} detected")
            else:
                # Update existing face info
                face_data = self.faces[face_id]
                
                # Update face info
                face_data['last_seen'] = current_time
                face_data['position'] = (center_x, center_y)
                face_data['rect'] = (x, y, w, h)
                
                # If face disappeared and reappeared, update total time correctly
                if 'disappeared' in face_data and face_data['disappeared']:
                    disappeared_duration = current_time - face_data['disappeared_time']
                    
                    # If face was gone for less than 3 seconds, count that time
                    if disappeared_duration < 3.0:
                        face_data['total_time'] += disappeared_duration
                    
                    face_data['disappeared'] = False
                
                # Update embedding with exponential moving average for stability
                if self.advanced_recognition and i < len(embeddings_list) and embeddings_list[i] is not None:
                    if 'embedding' not in face_data or face_data['embedding'] is None:
                        face_data['embedding'] = embeddings_list[i]
                    else:
                        # Blend new embedding with existing (smoothing)
                        alpha = 0.2  # weight of new embedding (0-1)
                        face_data['embedding'] = alpha * embeddings_list[i] + (1 - alpha) * face_data['embedding']
                
                # Update total time
                face_data['total_time'] = current_time - face_data['first_seen'] - face_data.get('missed_time', 0)
                
                # Check if face has been present for too long
                if face_data['total_time'] > self.time_limit and not face_data['alert']:
                    face_data['alert'] = True
                    print(f"Alert! Face {face_id} present for {face_data['total_time']:.1f} seconds (> {self.time_limit}s)")
            
            # Always display duration in debug mode or if there's an alert
            face_data = self.faces[face_id]
            duration = face_data['total_time']
            alert_status = face_data['alert']
            
            # Choose text and color based on alert status
            if self.debug_mode or alert_status:
                text = f"ALERT! {duration:.1f}s" if alert_status else f"Time: {duration:.1f}s"
                color = (0, 0, 255) if alert_status else (0, 255, 0)
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw a thicker red rectangle for faces with alerts
                if alert_status:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # Handle faces that are no longer visible
        face_ids = list(self.faces.keys())
        for face_id in face_ids:
            if face_id not in current_faces:
                face_data = self.faces[face_id]
                
                # Mark face as disappeared if it wasn't already
                if not face_data.get('disappeared', False):
                    face_data['disappeared'] = True
                    face_data['disappeared_time'] = current_time
                
                # Calculate time since last seen
                time_gone = current_time - face_data['last_seen']
                
                # Remove face if it has been gone for too long
                if time_gone > 10.0:  # 10 seconds threshold for removal
                    # Only report if this face had triggered an alert
                    if face_data['alert'] and self.debug_mode:
                        print(f"Face {face_id} no longer tracked (gone for {time_gone:.1f}s)")
                    del self.faces[face_id]
        
        return self.faces
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_time_limit(self, time_limit):
        self.time_limit = time_limit
    
    def set_debug_mode(self, debug_mode):
        self.debug_mode = debug_mode