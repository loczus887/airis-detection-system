import cv2
import numpy as np
import time
import os
import threading
from collections import deque

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.debug_mode = False
        
        # Lista obiektów uważanych za niebezpieczne z dostosowanymi progami
        self.dangerous_objects = {
            "knife": 0.4,
            "scissors": 0.3,
            "fork": 0.3,
            "baseball bat": 0.4,
            "bottle": 0.5,
            "wine glass": 0.5,
            "cell phone": 0.5,
            "gun": 0.4,
            "rifle": 0.4,
            "pistol": 0.4,
            "axe": 0.4,
            "hammer": 0.4,
        }
        
        # Śledzenie obiektów między klatkami
        self.object_persistence = {}
        self.last_dangerous_objects = []
        
        # Historia detekcji dla uśredniania wyników
        self.detection_history = deque(maxlen=5)
        
        # Zmniejszona częstotliwość detekcji dla lepszej wydajności
        self.detection_interval = 5  # Wykonuj pełną detekcję co 5 klatek
        self.frame_count = 0
        
        # Bufor do przetwarzania w tle
        self.processing_frame = None
        self.processing_lock = threading.Lock()
        self.processing_thread = None
        self.processing_active = False
        
        # Wyniki ostatniej detekcji
        self.last_detection_results = {'objects': [], 'dangerous_objects': [], 'boxes': [], 'confidences': [], 'class_ids': []}
        
        # Do symulacji detekcji w przypadku braku modelu
        self.last_detection_time = 0
        
        # Inicjalizacja modelu YOLO
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicjalizacja modelu YOLO"""
        # Ścieżki do plików modelu - najpierw próbujemy tiny modele, potem większe
        model_paths = [
            ("yolov4-tiny.weights", "yolov4-tiny.cfg"),
            ("yolov3-tiny.weights", "yolov3-tiny.cfg"),
            ("yolov4.weights", "yolov4.cfg"),
            ("yolov3.weights", "yolov3.cfg"),
        ]
        
        self.net = None
        
        # Próbuj załadować kolejne modele, aż jeden zadziała
        for weights_path, config_path in model_paths:
            try:
                if os.path.exists(weights_path) and os.path.exists(config_path):
                    self.net = cv2.dnn.readNet(weights_path, config_path)
                    print(f"Loaded YOLO model from {weights_path} and {config_path}")
                    break
            except Exception as e:
                print(f"Failed to load {weights_path}: {e}")
        
        # Jeśli nie udało się załadować żadnego modelu, użyj symulacji
        if self.net is None:
            print("WARNING: No YOLO model loaded. Using simulated detection instead.")
            return
        
        # Sprawdź dostępność CUDA
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA for object detection")
            else:
                print("CUDA not available, using CPU for detection")
                # Optymalizacja dla CPU
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"Error configuring backend: {e}")
            print("Using default settings")
        
        # Załaduj nazwy klas
        try:
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.classes)} object classes")
        except:
            print("Warning: Failed to load coco.names, using limited class list")
            # Podstawowa lista klas COCO
            self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
                           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                           "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
                           "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
                           "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        # Uzyskaj nazwy warstw wyjściowych
        try:
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        except Exception as e:
            print(f"Error getting output layers: {e}")
            self.output_layers = []
        
        # Rozpocznij wątek detekcji w tle jeżeli model został załadowany
        if self.net is not None and len(self.output_layers) > 0:
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self._background_detection_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def enhance_for_small_objects(self, image):
        """Ulepszone przetwarzanie obrazu dla małych obiektów"""
        # Konwersja do przestrzeni kolorów LAB dla lepszego przetwarzania kolorów
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Zastosowanie CLAHE do kanału L dla lepszego kontrastu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Połączenie ulepszonego kanału L z oryginalnymi kanałami A i B
        limg = cv2.merge((cl, a, b))
        
        # Konwersja z powrotem do przestrzeni kolorów BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Opcjonalnie - wyostrzenie obrazu dla lepszego wykrywania krawędzi
        if self.debug_mode:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _background_detection_thread(self):
        """Wątek przetwarzania detekcji w tle"""
        while self.processing_active:
            # Sprawdź, czy jest ramka do przetworzenia
            with self.processing_lock:
                frame_to_process = self.processing_frame
                if frame_to_process is None:
                    time.sleep(0.01)  # Krótkie opóźnienie, aby nie obciążać CPU
                    continue
                
                self.processing_frame = None  # Zwolnij bufor
            
            # Wykonaj detekcję
            result = self._detect_objects(frame_to_process)
            
            # Zapisz wyniki
            if result:
                self.last_detection_results = result
                
                # Dodaj wyniki do historii dla uśredniania
                if 'dangerous_objects' in result and result['dangerous_objects']:
                    self.detection_history.append(result['dangerous_objects'])
                    
                    # Uaktualnij listę ostatnio wykrytych niebezpiecznych obiektów
                    all_detected = set()
                    for detection in self.detection_history:
                        all_detected.update(detection)
                    
                    self.last_dangerous_objects = list(all_detected)
    
    def _detect_objects(self, frame):
        """Wykonaj detekcję obiektów"""
        # Jeśli nie załadowano modelu, użyj symulowanej detekcji
        if self.net is None or not self.output_layers:
            return self._simulated_detection(frame)
        
        # Inicjalizacja wartości zwracanych
        detected_objects = []
        dangerous_detected = []
        
        try:
            # Przygotowanie klatki
            # Zmniejszenie rozmiaru dla lepszej wydajności
            small_frame = cv2.resize(frame, (416, 416))
            
            # Ulepszenie przetwarzania wstępnego dla małych obiektów
            if self.debug_mode:
                enhanced_frame = self.enhance_for_small_objects(small_frame)
            else:
                enhanced_frame = small_frame
            
            height, width, channels = enhanced_frame.shape
            
            # Tworzenie bloba dla sieci neuronowej
            blob = cv2.dnn.blobFromImage(enhanced_frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
            
            # Przetwarzanie przez sieć
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Listy do przechowywania wyników detekcji
            class_ids = []
            confidences = []
            boxes = []
            
            # Analiza wyników detekcji
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Pobierz etykietę klasy
                    if class_id < len(self.classes):
                        label = self.classes[class_id].lower()
                        
                        # Ustal odpowiedni próg dla tego obiektu
                        threshold = self.dangerous_objects.get(label, self.confidence_threshold)
                        
                        # Jeśli pewność detekcji przekracza próg
                        if confidence > threshold:
                            # Oblicz koordynaty prostokąta
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Lewy górny róg
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            # Dodaj do list
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            # Zastosuj non-maximum suppression, aby uniknąć wielokrotnych detekcji
            if boxes:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
                
                # Przetwarzanie wybranych detekcji
                object_this_frame = {}
                
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = self.classes[class_ids[i]].lower()
                        confidence = confidences[i]
                        
                        # Skaluj współrzędne do oryginalnego rozmiaru klatki
                        scale_x = frame.shape[1] / float(width)
                        scale_y = frame.shape[0] / float(height)
                        x = int(x * scale_x)
                        y = int(y * scale_y)
                        w = int(w * scale_x)
                        h = int(h * scale_y)
                        
                        # Dodaj wykryty obiekt do listy
                        detected_objects.append(label)
                        
                        # Sprawdź, czy to niebezpieczny obiekt
                        if label in self.dangerous_objects:
                            # Stwórz klucz przestrzenny dla obiektu
                            object_key = f"{label}_{x//50}_{y//50}"
                            object_this_frame[object_key] = True
                            
                            # Aktualizuj trwałość obiektu
                            if object_key in self.object_persistence:
                                self.object_persistence[object_key] += 1
                            else:
                                self.object_persistence[object_key] = 1
                            
                            # Rozważ obiekty, które zostały wykryte wielokrotnie
                            required_detections = 1 if self.debug_mode else 2
                            
                            if self.object_persistence[object_key] >= required_detections:
                                dangerous_detected.append(label)
                
                # Usuń obiekty, które nie są już wykrywane
                keys_to_remove = []
                for key in self.object_persistence:
                    if key not in object_this_frame:
                        self.object_persistence[key] -= 1
                        if self.object_persistence[key] <= 0:
                            keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.object_persistence[key]
                
                # Aktualizuj listę ostatnio wykrytych niebezpiecznych obiektów
                if dangerous_detected:
                    self.last_dangerous_objects = list(set(dangerous_detected))
            
            # Zwróć wyniki detekcji
            return {
                'objects': detected_objects,
                'dangerous_objects': dangerous_detected,
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in object detection: {e}")
            # Jeśli wystąpi błąd, użyj symulacji jako metody awaryjnej
            return self._simulated_detection(frame)
    
    def process(self, frame):
        """Przetwórz klatkę - z obsługą przetwarzania w tle"""
        self.frame_count += 1
        
        # Pobieram ostatnie wyniki detekcji
        results = self.last_detection_results
        
        # Co kilka klatek, zakolejkuj nową ramkę do przetworzenia
        if self.frame_count % self.detection_interval == 0:
            # Dodaj ramkę do przetworzenia w tle
            with self.processing_lock:
                self.processing_frame = frame.copy()
        
        # Jeśli nie mamy jeszcze wyników lub jesteśmy w trybie symulacji, wykonaj symulację
        if not results.get('objects') and not results.get('dangerous_objects'):
            if self.net is None:
                return self._simulated_detection(frame)
        
        # Rysowanie prostokątów na podstawie ostatnich wyników
        if 'boxes' in results and 'class_ids' in results and 'confidences' in results:
            boxes = results['boxes']
            class_ids = results['class_ids']
            confidences = results['confidences']
            
            # Przetwarzanie wybranych detekcji
            for i in range(len(boxes)):
                if i < len(class_ids) and i < len(confidences):
                    x, y, w, h = boxes[i]
                    
                    # Skaluj współrzędne do oryginalnego rozmiaru klatki
                    scale_x = frame.shape[1] / 416.0
                    scale_y = frame.shape[0] / 416.0
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    
                    # Sprawdź, czy indeks jest prawidłowy
                    if class_ids[i] < len(self.classes):
                        label = self.classes[class_ids[i]].lower()
                        confidence = confidences[i]
                        
                        # Sprawdź, czy to niebezpieczny obiekt
                        if label in self.dangerous_objects:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # W trybie debug rysuj wszystkie obiekty
                        elif self.debug_mode:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Wyświetl alert, jeśli znaleziono niebezpieczne obiekty
        if results.get('dangerous_objects'):
            alert_text = f"⚠️ NIEBEZPIECZEŃSTWO: wykryto {', '.join(set(results['dangerous_objects']))}!"
            cv2.putText(frame, alert_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.last_dangerous_objects and self.frame_count % 30 == 0:
            # Okresowe przypominanie o ostatnich detekcjach
            cv2.putText(frame, f"Ostatnio wykryto: {', '.join(set(self.last_dangerous_objects))}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        return results
    
    def _simulated_detection(self, frame):
        """Symulowana detekcja obiektów (używana jako metoda awaryjna)"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        current_time = time.time()
        
        # Inicjalizacja wartości zwracanych
        detected_objects = []
        dangerous_detected = []
        boxes = []
        confidences = []
        class_ids = []
        
        # Symulacja regularnego wykrywania obiektów
        # Co 5 sekund wykryj normalny obiekt
        if int(current_time) % 5 == 0:
            # Tylko raz na sekundę
            if current_time - self.last_detection_time >= 1:
                self.last_detection_time = current_time
                detected_objects.append("bottle")
                
                # Dodaj dane do symulowanych list
                x1, y1 = center_x - 50, center_y - 50
                x2, y2 = center_x + 50, center_y + 50
                boxes.append([x1, y1, 100, 100])
                confidences.append(0.8)
                bottle_class_id = self.classes.index("bottle") if "bottle" in self.classes else 39
                class_ids.append(bottle_class_id)
                
                # Zielony prostokąt dla normalnego obiektu
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if self.debug_mode:
                    cv2.putText(frame, "bottle 0.80", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Co 10 sekund wykryj niebezpieczny obiekt
        if int(current_time) % 10 == 0:
            # Tylko raz na sekundę
            if current_time - self.last_detection_time >= 1:
                self.last_detection_time = current_time
                
                # Wybierz niebezpieczny obiekt na podstawie czasu
                dangerous_objects_list = list(self.dangerous_objects.keys())
                index = (int(current_time) // 10) % len(dangerous_objects_list)
                dangerous_item = dangerous_objects_list[index]
                
                dangerous_detected.append(dangerous_item)
                detected_objects.append(dangerous_item)
                
                # Dodaj dane do symulowanych list
                x1, y1 = center_x - 100, center_y - 30
                x2, y2 = center_x + 100, center_y + 70
                boxes.append([x1, y1, 200, 100])
                confidences.append(0.7)
                class_id = self.classes.index(dangerous_item) if dangerous_item in self.classes else 0
                class_ids.append(class_id)
                
                # Czerwony prostokąt dla niebezpiecznego obiektu
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Etykieta dla niebezpiecznych obiektów
                cv2.putText(frame, f"{dangerous_item} 0.70", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Wyświetl alert
                alert_text = f"⚠️ NIEBEZPIECZEŃSTWO: wykryto {dangerous_item}!"
                cv2.putText(frame, alert_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Aktualizuj listę ostatnio wykrytych niebezpiecznych obiektów
                self.last_dangerous_objects = [dangerous_item]
        
        return {
            'objects': detected_objects,
            'dangerous_objects': dangerous_detected,
            'boxes': boxes,
            'confidences': confidences, 
            'class_ids': class_ids
        }
    
    def set_confidence_threshold(self, threshold):
        """Ustaw próg pewności dla wszystkich obiektów"""
        self.confidence_threshold = threshold
    
    def set_debug_mode(self, debug_mode):
        """Włącz/wyłącz tryb debugowania"""
        self.debug_mode = debug_mode
        
        # W trybie debug, zwiększamy częstotliwość detekcji
        self.detection_interval = 3 if debug_mode else 5