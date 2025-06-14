import cv2
import numpy as np
import time
import threading
import requests  # Dodaj ten import
from urllib.parse import urlparse
import socket

class CameraManager:
    def __init__(self):
        self.capture = None
        self.camera_index = 0
        self.available_cameras = self._get_available_cameras()
        self.ip_camera_url = None
        self.frame_buffer = None
        self.last_frame_time = 0
        self.frame_rate = 30
        self.resolution = (640, 480)  # Domyślna rozdzielczość
        
        # Opcje przetwarzania
        self.enable_preprocessing = False
    
        # Parametry dla ESP32-CAM
        self.connection_timeout = 10.0  # Timeout dla połączenia (sekundy)
        self.reconnect_attempts = 3     # Liczba prób ponownego połączenia
        self.reconnect_delay = 2.0      # Opóźnienie między próbami (sekundy)
        
        # Dodane flagi dla ESP32-CAM
        self.is_esp32_camera = False    # Flaga określająca czy to kamera ESP32
        self.esp32_capture_mode = False # Tryb pobierania pojedynczych zdjęć
        self.frame_lock = threading.Lock() # Blokada dla bezpiecznej aktualizacji bufora

    def _get_available_cameras(self):
        """Wykryj dostępne kamery w systemie"""
        available = []
        for i in range(5):  # Sprawdź pierwsze 5 indeksów kamer
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        
        print(f"Wykryto {len(available)} dostępne kamery: {available}")
        return available
    
    
    def _is_host_reachable(self, host, port):
        """Sprawdź, czy host jest osiągalny przed próbą połączenia"""
        try:
            # Spróbuj otworzyć połączenie TCP
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)  # 2 sekundy timeout
            s.connect((host, port))
            s.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False
    
    def start(self, camera_index=0):
        """Rozpocznij przechwytywanie z kamery"""
        try:
            self.camera_index = camera_index
            # Zwolnij bieżącą kamerę, jeśli jest otwarta
            if self.capture and self.capture.isOpened():
                self.capture.release()
            
            # Otwórz nową kamerę
            self.capture = cv2.VideoCapture(camera_index)
            
            # Ustaw rozdzielczość i FPS dla lepszej wydajności
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            # Ustaw bufor kamery na minimalne opóźnienie
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.capture.isOpened():
                # Pobierz faktyczną rozdzielczość i FPS
                actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
                
                print(f"Kamera {camera_index} uruchomiona: {actual_width}x{actual_height} @ {actual_fps} FPS")
                return True
            else:
                print(f"Nie można otworzyć kamery {camera_index}")
                return False
        except Exception as e:
            print(f"Błąd uruchamiania kamery: {e}")
            return False
    
    def get_frame(self):
        """Pobierz klatkę z kamery z opcjonalnym przetwarzaniem wstępnym"""
        # Dla trybu ESP32-CAM capture
        if hasattr(self, 'is_esp32_camera') and self.is_esp32_camera and hasattr(self, 'esp32_capture_mode') and self.esp32_capture_mode:
            try:
                # Kontrola częstotliwości klatek
                current_time = time.time()
                frame_interval = 1.0 / self.frame_rate
                
                # Pobieraj nową klatkę tylko gdy upłynął odpowiedni czas
                if current_time - self.last_frame_time >= frame_interval:
                    # Pobierz zdjęcie z URL capture
                    try:
                        response = requests.get(self.ip_camera_url, timeout=2)
                        if response.status_code == 200:
                            # Konwertuj odpowiedź na obraz numpy
                            nparr = np.frombuffer(response.content, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Aktualizuj bufor klatek
                                with self.frame_lock:
                                    self.frame_buffer = frame
                                    self.last_frame_time = current_time
                            else:
                                print("Nie można zdekodować obrazu z odpowiedzi")
                        else:
                            print(f"Błąd pobierania klatki: kod {response.status_code}")
                    except Exception as e:
                        print(f"Wyjątek przy pobieraniu klatki: {e}")
                
                # Opcjonalne przetwarzanie wstępne
                if self.enable_preprocessing and self.frame_buffer is not None:
                    processed_frame = self._preprocess_frame(self.frame_buffer.copy())
                    return processed_frame
                
                return self.frame_buffer
            except Exception as e:
                print(f"Błąd pobierania klatki z ESP32-CAM: {e}")
                return self.frame_buffer
        
        # Standardowe pobieranie klatki z kamery
        if not self.capture or not self.capture.isOpened():
            return None
        
        # Kontrola częstotliwości klatek
        current_time = time.time()
        frame_interval = 1.0 / self.frame_rate
        
        # Jeśli bufor jest pełny, poczekaj przed pobraniem nowej klatki
        if self.last_frame_time + frame_interval > current_time and self.frame_buffer is not None:
            return self.frame_buffer
        
        # Pobierz nową klatkę
        ret, frame = self.capture.read()
        
        # Obsługa błędów dla kamery
        if not ret:
            print("Błąd odczytu klatki z kamery")
            if self.frame_buffer is not None:
                return self.frame_buffer
            return None
        
        # Aktualizuj czas ostatniej klatki
        self.last_frame_time = current_time
        
        # Opcjonalne przetwarzanie wstępne
        if self.enable_preprocessing:
            frame = self._preprocess_frame(frame)
        
        # Aktualizuj bufor klatek
        self.frame_buffer = frame
        
        return frame
    
    def _preprocess_frame(self, frame):
        """Przetwarzanie wstępne obrazu dla lepszej jakości detekcji"""
        try:
            # Redukcja szumów
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # Poprawa kontrastu i jasności
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Zastosowanie CLAHE do kanału L (jasność)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Połączenie ulepszonego kanału L z oryginalnymi kanałami A i B
            limg = cv2.merge((cl, a, b))
            
            # Konwersja z powrotem do przestrzeni kolorów BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            print(f"Błąd przetwarzania wstępnego: {e}")
            return frame
    
    def toggle_camera(self):
        """Przełączanie między dostępnymi kamerami"""
        if not self.available_cameras:
            print("Brak dostępnych kamer")
            return False
        
        # Zamknij bieżącą kamerę
        if self.capture:
            self.capture.release()
        
        # Znajdź następny indeks kamery
        if self.camera_index in self.available_cameras:
            current_idx = self.available_cameras.index(self.camera_index)
            next_idx = (current_idx + 1) % len(self.available_cameras)
            self.camera_index = self.available_cameras[next_idx]
        else:
            self.camera_index = self.available_cameras[0]
        
        # Otwórz nową kamerę
        return self.start(self.camera_index)
    
    def connect_to_ip_camera(self, ip, port=80, streaming_port=81):
        """Połączenie z kamerą IP ESP32-CAM"""
        try:
            print(f"Próba znalezienia dostępnych portów na {ip}...")
            
            # Sprawdź najpierw czy host jest w ogóle dostępny (na standardowym porcie HTTP)
            if not self._is_host_reachable(ip, port):
                print(f"Host {ip}:{port} (HTTP) nie jest dostępny")
                print("Sprawdzam alternatywne porty...")
                # Sprawdź alternatywne porty
                alternative_ports = [8080, 8000, 8081, 80, 81]
                available_ports = []
                
                for alt_port in alternative_ports:
                    if self._is_host_reachable(ip, alt_port):
                        print(f"Port {alt_port} jest dostępny na {ip}")
                        available_ports.append(alt_port)
                
                if not available_ports:
                    print(f"Nie znaleziono żadnych dostępnych portów na {ip}")
                    return False
                
                # Użyj pierwszego dostępnego portu dla HTTP
                port = available_ports[0]
                print(f"Używam portu {port} dla HTTP")
                
                # Spróbuj użyć kolejnego dostępnego portu (jeśli istnieje) dla strumienia
                if len(available_ports) > 1:
                    streaming_port = available_ports[1]
                else:
                    streaming_port = port
                    
                print(f"Używam portu {streaming_port} dla strumienia")
            
            # Sprawdź możliwe ścieżki dla strumienia
            stream_paths = ['/stream', '/video', '/mjpeg', '/']
            
            # Próbuj różne kombinacje portów i ścieżek
            for s_port in [streaming_port, port]:
                for path in stream_paths:
                    self.ip_camera_url = f"http://{ip}:{s_port}{path}"
                    print(f"Próba połączenia z: {self.ip_camera_url}")
                    
                    # Zamknij bieżącą kamerę, jeśli jest otwarta
                    if self.capture and self.capture.isOpened():
                        self.capture.release()
                    
                    # Otwórz kamerę IP
                    self.capture = cv2.VideoCapture(self.ip_camera_url, cv2.CAP_FFMPEG)
                    
                    # Sprawdź, czy udało się otworzyć
                    if self.capture.isOpened():
                        # Pobierz testową klatkę, aby upewnić się, że połączenie działa
                        ret, test_frame = self.capture.read()
                        if ret and test_frame is not None:
                            # Udało się połączyć i odczytać klatkę
                            self.is_esp32_camera = True
                            self.esp32_capture_mode = False
                            print(f"Pomyślnie połączono ze strumieniem ESP32-CAM: {self.ip_camera_url}")
                            return True
                        else:
                            self.capture.release()
                            print(f"Połączono, ale nie można odczytać klatki z: {self.ip_camera_url}")
            
            # Jeśli strumień nie zadziałał, spróbuj trybu capture
            capture_paths = ['/capture', '/still', '/jpg', '/photo']
            for c_port in [port, streaming_port]:
                for path in capture_paths:
                    capture_url = f"http://{ip}:{c_port}{path}"
                    print(f"Próba pobrania pojedynczej klatki: {capture_url}")
                    
                    try:
                        response = requests.get(capture_url, timeout=5)
                        if response.status_code == 200:
                            # Sprawdź czy odpowiedź zawiera obraz
                            if 'image' in response.headers.get('content-type', ''):
                                import numpy as np
                                nparr = np.frombuffer(response.content, np.uint8)
                                self.frame_buffer = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if self.frame_buffer is not None:
                                    self.ip_camera_url = capture_url
                                    self.is_esp32_camera = True
                                    self.esp32_capture_mode = True
                                    print(f"Połączono w trybie pojedynczych zdjęć: {capture_url}")
                                    return True
                    except Exception as e:
                        print(f"Błąd pobierania z {capture_url}: {e}")
            
            print("Nie udało się znaleźć działającego strumienia lub trybu capture")
            return False
            
        except Exception as e:
            print(f"Błąd łączenia z kamerą ESP32-CAM: {e}")
            return False
    
    def set_resolution(self, width, height):
        """Ustaw rozdzielczość kamery"""
        self.resolution = (width, height)
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Sprawdź, czy zmiana rozdzielczości powiodła się
            actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"Rozdzielczość kamery ustawiona na: {actual_width}x{actual_height}")
    
    def set_frame_rate(self, fps):
        """Ustaw liczbę klatek na sekundę"""
        self.frame_rate = fps
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            
            # Sprawdź, czy zmiana FPS powiodła się
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            print(f"Liczba klatek na sekundę ustawiona na: {actual_fps}")
    
    def enable_image_preprocessing(self, enable=True):
        """Włącz/wyłącz przetwarzanie wstępne obrazu"""
        self.enable_preprocessing = enable
    
    def release(self):
        """Zwolnij zasoby kamery"""
        if self.capture:
            self.capture.release()
            print("Kamera zwolniona")