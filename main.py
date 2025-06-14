from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup

import cv2
import time
import os
import threading

# Import detection modules
from detection.face_detector import FaceDetector
from detection.object_detector import ObjectDetector
from utils.camera_manager import CameraManager


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            # Initialize components
        self.camera_manager = CameraManager()
        self.face_detector = None
        self.object_detector = None
        
        # App state
        self.is_capturing = False
        self.debug_mode = False
        self.alerts = []
        
        # Settings
        self.settings = {
            'enable_face_detection': True,
            'enable_object_detection': True,
            'time_limit': 10,  # seconds before face alert
            'face_threshold': 0.6,
            'object_confidence': 0.5,
            # ESP32-CAM settings
            'esp32_ip': '192.168.0.249',
            'esp32_port': 80
        }

        self.layout = BoxLayout(orientation='vertical')
        
        # Simple title label
        self.title = Label(text="Security Monitoring", size_hint_y=0.1, font_size=20)
        self.layout.add_widget(self.title)
        
        # Camera view
        self.camera_view = Image(size_hint_y=0.7)
        self.layout.add_widget(self.camera_view)
        
        # Button layout
        self.button_layout = GridLayout(cols=3, size_hint_y=0.1, spacing=10, padding=10)
        
        # Add buttons
        self.start_btn = Button(text="Start Camera")
        self.start_btn.bind(on_press=self.start_camera_callback)  # Fixed method name
        self.button_layout.add_widget(self.start_btn)
        
        self.stop_btn = Button(text="Stop Camera")
        self.stop_btn.bind(on_press=self.stop_camera_callback)  # Fixed method name
        self.button_layout.add_widget(self.stop_btn)
        
        self.toggle_camera_btn = Button(text="Toggle Camera")
        self.toggle_camera_btn.bind(on_press=self.toggle_camera_callback)  # Fixed method name
        self.button_layout.add_widget(self.toggle_camera_btn)
        
        self.connect_glasses_btn = Button(text="Connect ESP32-CAM")
        self.connect_glasses_btn.bind(on_press=self.connect_to_glasses_callback)  # Zachowujemy oryginalną nazwę metody
        self.button_layout.add_widget(self.connect_glasses_btn)

        self.debug_btn = Button(text="Debug Mode")
        self.debug_btn.bind(on_press=self.toggle_debug_callback)  # Fixed method name
        self.button_layout.add_widget(self.debug_btn)
        
        self.settings_btn = Button(text="Settings")
        self.settings_btn.bind(on_press=self.go_to_settings_callback)  # Fixed method name
        self.button_layout.add_widget(self.settings_btn)
        
        self.alerts_btn = Button(text="Alerts")
        self.alerts_btn.bind(on_press=self.go_to_alerts_callback)  # Fixed method name
        self.button_layout.add_widget(self.alerts_btn)

         
        # Dodaj przycisk skanujący kamery ESP32-CAM w sieci
        self.scan_esp32_btn = Button(text="Scan Network")
        self.scan_esp32_btn.bind(on_press=self.scan_network_callback)
        self.button_layout.add_widget(self.scan_esp32_btn)
        
        self.layout.add_widget(self.button_layout)

         # Status layout with connection info
        self.status_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        
        
        # Status label
        self.status_label = Label(text="System ready", size_hint_y=0.1)
        self.layout.add_widget(self.status_label)

         # Connection status for ESP32-CAM
        self.connection_label = Label(text="", size_hint_x=0.3, color=(0, 1, 0, 1))
        self.status_layout.add_widget(self.connection_label)
        
        self.layout.add_widget(self.status_layout)
        
        # Alert indicator (using a label for simplicity)
        self.alert_indicator = Label(
            text="⚠️ ALERT", 
            color=(1, 0, 0, 1), 
            size_hint_y=0.05,
            opacity=0
        )
        self.layout.add_widget(self.alert_indicator)
        
        self.add_widget(self.layout)

         # Start checking ESP32 connection status
        Clock.schedule_interval(self.update_connection_status, 1.0)
    
    def update_connection_status(self, dt):
        """Update the connection status indicator"""
        app = App.get_running_app()
        if app.camera_manager and app.camera_manager.is_esp32_camera:
            self.connection_label.text = f"ESP32: {app.settings.get('esp32_ip', '')}"
            self.connection_label.color = (0, 1, 0, 1)  # Zielony dla aktywnego połączenia
        else:
            if app.settings.get('use_esp32', False):
                self.connection_label.text = "ESP32: Disconnected"
                self.connection_label.color = (1, 0, 0, 1)  # Czerwony dla braku połączenia
            else:
                self.connection_label.text = ""  # Puste gdy nie używamy ESP32
    
    # Fixed method names with _callback suffix to avoid confusion
    def start_camera_callback(self, instance):
        app = App.get_running_app()
        app.start_camera()
        self.status_label.text = "Camera started"
    
    def stop_camera_callback(self, instance):
        app = App.get_running_app()
        app.stop_camera()
        self.status_label.text = "Camera stopped"
    
    def toggle_camera_callback(self, instance):
        app = App.get_running_app()
        if app.toggle_camera():
            if app.camera_manager.is_esp32_camera:
                self.status_label.text = "Switched to ESP32-CAM"
            else:
                self.status_label.text = f"Switched to camera {app.camera_manager.camera_index}"
        else:
            self.status_label.text = "Failed to switch camera"
    
    def connect_to_glasses_callback(self, instance):
        """Ta metoda musi zachować oryginalną nazwę z uwagi na wiązanie przycisku"""
        app = App.get_running_app()
        
        # Zmień tekst przycisku podczas łączenia
        self.connect_glasses_btn.text = "Connecting..."
        self.connect_glasses_btn.disabled = True
        
        # Pokaż informacje w etykiecie statusu
        self.status_label.text = f"Connecting to ESP32-CAM at {app.settings.get('esp32_ip', '...')}..."
        
        # Próba połączenia z ESP32-CAM
        if app.connect_to_glasses():
            # Połączenie zainicjowane, przycisk zostanie zaktualizowany przez callback
            pass
        else:
            # Natychmiastowy błąd
            self.status_label.text = "Failed to connect to ESP32-CAM"
            self.connect_glasses_btn.text = "Connect ESP32-CAM"
            self.connect_glasses_btn.disabled = False
        
        # Zresetuj przycisk po 5 sekundach bez względu na wynik
        def reset_button(dt):
            self.connect_glasses_btn.text = "Connect ESP32-CAM"
            self.connect_glasses_btn.disabled = False
            
        Clock.schedule_once(reset_button, 5)

    
    def toggle_debug_callback(self, instance):
        app = App.get_running_app()
        app.toggle_debug_mode()
        if app.debug_mode:
            self.status_label.text = "Debug mode ON"
        else:
            self.status_label.text = "Debug mode OFF"
    
    def go_to_settings_callback(self, instance):
        app = App.get_running_app()
        app.settings_screen.update_ui()  # Update settings UI before showing
        self.manager.current = 'settings'
    
    def go_to_alerts_callback(self, instance):
        app = App.get_running_app()
        app.alerts_screen.update_alerts(app.alerts)
        self.hide_alert_indicator()
        self.manager.current = 'alerts'
    
    def scan_network_callback(self, instance):
        """Scan network for ESP32-CAM devices"""
        app = App.get_running_app()
        
        # Zmień tekst przycisku i wyłącz go podczas skanowania
        self.scan_esp32_btn.text = "Scanning..."
        self.scan_esp32_btn.disabled = True
        self.status_label.text = "Scanning network for ESP32-CAM devices..."
        
        # Wykonaj skanowanie w osobnym wątku
        def scan_thread():
            # Uzyskaj bieżący adres IP
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Uzyskaj bazowy adres sieci
            ip_parts = local_ip.split('.')
            base_ip = '.'.join(ip_parts[0:3]) + '.'
            
            # Lista znalezionych urządzeń ESP32
            found_devices = []
            
            # Skanuj pierwsze 20 adresów w sieci
            for i in range(1, 20):
                ip_to_check = f"{base_ip}{i}"
                
                # Sprawdź, czy urządzenie odpowiada na port 80
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)  # Krótki timeout
                
                try:
                    result = sock.connect_ex((ip_to_check, 80))
                    if result == 0:
                        # Port jest otwarty, spróbuj sprawdzić czy to ESP32-CAM
                        try:
                            import requests
                            response = requests.get(f"http://{ip_to_check}", timeout=1)
                            if "ESP32" in response.text or response.status_code == 200:
                                found_devices.append(ip_to_check)
                        except:
                            # Jeśli nie możemy potwierdzić, ale port jest otwarty, dodaj jako potencjalne
                            found_devices.append(ip_to_check)
                except:
                    pass
                finally:
                    sock.close()
            
            # Aktualizuj UI w głównym wątku
            def update_ui(dt):
                self.scan_esp32_btn.text = "Scan Network"
                self.scan_esp32_btn.disabled = False
                
                if found_devices:
                    # Jeśli znaleziono urządzenia, pokaż je i ustaw pierwsze jako domyślne
                    app.settings['esp32_ip'] = found_devices[0]
                    self.status_label.text = f"Found ESP32-CAM at: {', '.join(found_devices)}"
                    
                    # Pokaż okno dialogowe z wyborem urządzenia
                    if len(found_devices) > 1:
                        self.show_esp32_selection_dialog(found_devices)
                else:
                    self.status_label.text = "No ESP32-CAM devices found on network"
            
            # Zaktualizuj UI
            Clock.schedule_once(update_ui, 0)
        
        # Uruchom skanowanie w tle
        import threading
        t = threading.Thread(target=scan_thread)
        t.daemon = True
        t.start()
    
    def show_esp32_selection_dialog(self, device_list):
        """Pokaż dialog wyboru ESP32-CAM"""
        app = App.get_running_app()
        
        # Utwórz layout dla dialogu
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text="Select ESP32-CAM device:"))
        
        # Utwórz przyciski dla każdego urządzenia
        buttons = []
        for device in device_list:
            btn = Button(text=device)
            buttons.append(btn)
            content.add_widget(btn)
        
        # Utwórz popup
        from kivy.uix.popup import Popup
        popup = Popup(title='ESP32-CAM Devices', content=content, size_hint=(0.8, 0.8))
        
        # Dodaj obsługę kliknięć przycisków
        for btn in buttons:
            def select_device(instance, device=btn.text):
                app.settings['esp32_ip'] = device
                self.status_label.text = f"Selected ESP32-CAM at {device}"
                popup.dismiss()
            
            btn.bind(on_press=select_device)
        
        # Pokaż popup
        popup.open()

    
    def show_alert_indicator(self):
        self.alert_indicator.opacity = 1
    
    def hide_alert_indicator(self):
        self.alert_indicator.opacity = 0


class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        
        # Title
        self.title = Label(text="Settings", size_hint_y=0.1, font_size=20)
        self.layout.add_widget(self.title)
        
        # Settings options
        self.settings_layout = GridLayout(cols=2, size_hint_y=0.7, spacing=10, padding=10)
        
        # W klasie SettingsScreen - dodaj do __init__
        self.settings_layout.add_widget(Label(text="ESP32-CAM IP:"))
        self.ip_address_input = TextInput(text="192.168.0.249", multiline=False)
        self.settings_layout.add_widget(self.ip_address_input)

        self.settings_layout.add_widget(Label(text="ESP32-CAM Port:"))
        self.port_input = TextInput(text="80", multiline=False, input_filter="int")
        self.settings_layout.add_widget(self.port_input)

        # Face detection settings
        self.settings_layout.add_widget(Label(text="Face Detection:"))
        self.face_detection_btn = Button(text="ON")
        self.face_detection_btn.bind(on_press=self.toggle_face_detection)
        self.settings_layout.add_widget(self.face_detection_btn)
        
        # Object detection settings
        self.settings_layout.add_widget(Label(text="Object Detection:"))
        self.object_detection_btn = Button(text="ON")
        self.object_detection_btn.bind(on_press=self.toggle_object_detection)
        self.settings_layout.add_widget(self.object_detection_btn)
        
        # Time limit settings
        self.settings_layout.add_widget(Label(text="Face Alert Time (seconds):"))
        self.time_limit_input = TextInput(text="10", multiline=False, input_filter="int")
        self.settings_layout.add_widget(self.time_limit_input)
        
        # Face threshold settings
        self.settings_layout.add_widget(Label(text="Face Threshold:"))
        self.face_threshold_input = TextInput(text="0.6", multiline=False, input_filter="float")
        self.settings_layout.add_widget(self.face_threshold_input)
        
        # Object confidence settings
        self.settings_layout.add_widget(Label(text="Object Confidence:"))
        self.object_confidence_input = TextInput(text="0.5", multiline=False, input_filter="float")
        self.settings_layout.add_widget(self.object_confidence_input)
        
        self.layout.add_widget(self.settings_layout)
        
        # Button layout
        self.button_layout = BoxLayout(size_hint_y=0.2, spacing=10, padding=10)
        
        # Back button
        self.back_btn = Button(text="Back")
        self.back_btn.bind(on_press=self.go_back)
        self.button_layout.add_widget(self.back_btn)
        
        # Save button
        self.save_btn = Button(text="Save Settings")
        self.save_btn.bind(on_press=self.save_settings)
        self.button_layout.add_widget(self.save_btn)
        
        self.layout.add_widget(self.button_layout)
        
        self.add_widget(self.layout)
    
    def toggle_face_detection(self, instance):
        app = App.get_running_app()
        app.settings['enable_face_detection'] = not app.settings['enable_face_detection']
        self.face_detection_btn.text = "ON" if app.settings['enable_face_detection'] else "OFF"
    
    def toggle_object_detection(self, instance):
        app = App.get_running_app()
        app.settings['enable_object_detection'] = not app.settings['enable_object_detection']
        self.object_detection_btn.text = "ON" if app.settings['enable_object_detection'] else "OFF"
    
    def update_ui(self):
        """Update UI to reflect current settings"""
        app = App.get_running_app()
        self.face_detection_btn.text = "ON" if app.settings['enable_face_detection'] else "OFF"
        self.object_detection_btn.text = "ON" if app.settings['enable_object_detection'] else "OFF"
        self.time_limit_input.text = str(app.settings['time_limit'])
        self.face_threshold_input.text = str(app.settings['face_threshold'])
        self.object_confidence_input.text = str(app.settings['object_confidence'])
        
        # Dodaj ustawienia ESP32-CAM
        if 'esp32_ip' in app.settings:
            self.ip_address_input.text = app.settings['esp32_ip']
        if 'esp32_port' in app.settings:
            self.port_input.text = str(app.settings['esp32_port'])

    def go_back(self, instance):
        self.manager.current = 'main'
    
    def save_settings(self, instance):
        app = App.get_running_app()
        
        # Update settings from UI
        try:
            app.settings['time_limit'] = int(self.time_limit_input.text)
            app.settings['face_threshold'] = float(self.face_threshold_input.text)
            app.settings['object_confidence'] = float(self.object_confidence_input.text)
            
            # Zapisz ustawienia ESP32-CAM
            app.settings['esp32_ip'] = self.ip_address_input.text
            app.settings['esp32_port'] = int(self.port_input.text)
            
            # Apply settings to detectors
            if app.face_detector:
                app.face_detector.set_threshold(app.settings['face_threshold'])
                app.face_detector.set_time_limit(app.settings['time_limit'])
            
            if app.object_detector:
                app.object_detector.set_confidence_threshold(app.settings['object_confidence'])
            
            app.main_screen.status_label.text = "Settings saved"
        except ValueError:
            app.main_screen.status_label.text = "Invalid settings"
        
        # Return to main screen
        self.manager.current = 'main'


class AlertsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        
        # Title
        self.title = Label(text="Alerts", size_hint_y=0.1, font_size=20)
        self.layout.add_widget(self.title)
        
        # Alerts list in a scrollview
        self.scroll_view = ScrollView(size_hint_y=0.7)
        self.alerts_layout = GridLayout(cols=1, size_hint_y=None, spacing=5, padding=5)
        self.alerts_layout.bind(minimum_height=self.alerts_layout.setter('height'))
        self.scroll_view.add_widget(self.alerts_layout)
        self.layout.add_widget(self.scroll_view)
        
        # Button layout
        self.button_layout = BoxLayout(size_hint_y=0.2, spacing=10, padding=10)
        
        # Back button
        self.back_btn = Button(text="Back")
        self.back_btn.bind(on_press=self.go_back)
        self.button_layout.add_widget(self.back_btn)
        
        # Clear alerts button
        self.clear_btn = Button(text="Clear All Alerts")
        self.clear_btn.bind(on_press=self.clear_alerts)
        self.button_layout.add_widget(self.clear_btn)
        
        self.layout.add_widget(self.button_layout)
        
        self.add_widget(self.layout)
    
    def update_alerts(self, alerts):
        """Update the alerts list"""
        self.alerts_layout.clear_widgets()
        
        if not alerts:
            self.alerts_layout.add_widget(
                Label(text="No alerts", size_hint_y=None, height=44)
            )
            return
        
        for alert in alerts:
            # Używamy display_message zamiast message, jeśli istnieje
            alert_text = f"{alert['timestamp']} - {alert.get('display_message', alert['message'])}"
            alert_label = Label(
                text=alert_text, 
                size_hint_y=None, 
                height=44, 
                color=(1, 0, 0, 1),
                halign='left'
            )
            # Set text_size only after widget is added to get proper width
            alert_label.bind(width=lambda s, w: setattr(s, 'text_size', (w, None)))
            self.alerts_layout.add_widget(alert_label)
    
    def go_back(self, instance):
        self.manager.current = 'main'
    
    def clear_alerts(self, instance):
        app = App.get_running_app()
        app.alerts = []
        self.update_alerts([])
        app.main_screen.hide_alert_indicator()
        
        # If object detection is enabled, ensure it can trigger new alerts
        if app.object_detector:
            app.object_detector.last_dangerous_objects = []
        
        # Reset face alerts if face detection is enabled
        if app.face_detector:
            for face_id in list(app.face_detector.faces.keys()):
                app.face_detector.faces[face_id]['alert'] = False
                
        print("All alerts cleared")


class SecurityApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize components
        self.camera_manager = CameraManager()
        self.face_detector = None
        self.object_detector = None
        
        # App state
        self.is_capturing = False
        self.debug_mode = False
        self.alerts = []
        
        # Settings
        self.settings = {
            'enable_face_detection': True,
            'enable_object_detection': True,
            'time_limit': 10,  # seconds before face alert
            'face_threshold': 0.6,
            'object_confidence': 0.5,
            # ESP32-CAM settings
            'esp32_ip': '192.168.0.249',  # Domyślny adres IP dla ESP32-CAM
            'esp32_port': 80,             # Domyślny port HTTP
            'use_esp32': False            # Czy używać ESP32-CAM
        }
    
    def build(self):
        # Create screen manager
        self.sm = ScreenManager()
        
        # Create screens
        self.main_screen = MainScreen(name='main')
        self.settings_screen = SettingsScreen(name='settings')
        self.alerts_screen = AlertsScreen(name='alerts')
        
        # Add screens to manager
        self.sm.add_widget(self.main_screen)
        self.sm.add_widget(self.settings_screen)
        self.sm.add_widget(self.alerts_screen)
        
        return self.sm
    
    def on_start(self):
        """Initialize application on start"""
        # Initialize detectors
        self.initialize_detectors()
        
        # Check available cameras
        available_cameras = self.camera_manager.available_cameras
        print(f"Wykryto {len(available_cameras)} dostępne kamery: {available_cameras}")
        
        # Automatically start camera
        if available_cameras:
            self.start_camera()
    
    def initialize_detectors(self):
        """Initialize face and object detectors"""
        try:
            # Face detector
            self.face_detector = FaceDetector(
                threshold=self.settings['face_threshold'],
                time_limit=self.settings['time_limit']
            )
            
            # Object detector
            self.object_detector = ObjectDetector(
                confidence_threshold=self.settings['object_confidence']
            )
            
            print("Moduły detekcji zainicjalizowane")
        except Exception as e:
            print(f"Błąd inicjalizacji modułów detekcji: {e}")
    
    def start_camera(self):
        """Start the camera and processing"""
        if not self.is_capturing:
            # Sprawdź czy powinniśmy używać ESP32-CAM
            if self.settings.get('use_esp32', False):
                # Najpierw użyj zapisanych ustawień ESP32
                result = self.connect_to_glasses()
                if not result:
                    # Jeśli się nie powiodło, spróbuj zwykłej kamery
                    if not self.camera_manager.capture or not self.camera_manager.capture.isOpened():
                        self.camera_manager.start(0)
            else:
                # Start camera if not already started
                if not self.camera_manager.capture or not self.camera_manager.capture.isOpened():
                    self.camera_manager.start(0)
            
            # Start processing if camera is open
            if self.camera_manager.capture and self.camera_manager.capture.isOpened():
                self.is_capturing = True
                Clock.schedule_interval(self.process_frame, 1.0/30.0)  # 30 FPS
                print("Kamera uruchomiona")
            else:
                print("Nie można uruchomić kamery")
    
    def stop_camera(self):
        """Stop the camera and processing"""
        if self.is_capturing:
            Clock.unschedule(self.process_frame)
            self.is_capturing = False
            print("Kamera zatrzymana")
    
    def toggle_camera(self):
        """Toggle between available cameras"""
        result = self.camera_manager.toggle_camera()
        
        # Reset use_esp32 flag if we've toggled to a local camera
        if result and not self.camera_manager.is_esp32_camera:
            self.settings['use_esp32'] = False
        
        return result
    
    def connect_to_glasses(self):
        """Connect to ESP32-CAM camera"""
        # Pobierz ustawienia ESP32 z konfiguracji
        ip_address = self.settings.get('esp32_ip', '192.168.0.249')
        
        # Spróbuj najpierw bezpośrednio sprawdzić adres IP
        try:
            import socket
            print(f"Sprawdzam czy host {ip_address} jest dostępny...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            result = s.connect_ex((ip_address, 80))
            s.close()
            
            if result != 0:
                print(f"Host {ip_address} nie odpowiada na standardowym porcie HTTP (80)")
                print("Próbuję znaleźć ESP32-CAM w sieci lokalnej...")
                
                # Próba automatycznego wykrycia ESP32-CAM w sieci
                # Określ lokalny adres IP urządzenia
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                ip_base = '.'.join(local_ip.split('.')[:3]) + '.'
                
                print(f"Szukam urządzeń w sieci {ip_base}0/24...")
                
                # Sprawdź pierwsze 20 adresów w sieci
                for i in range(1, 20):
                    ip_to_check = f"{ip_base}{i}"
                    
                    # Pomiń własny adres IP
                    if ip_to_check == local_ip:
                        continue
                        
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.2)  # krótki timeout dla szybszego skanowania
                    result = s.connect_ex((ip_to_check, 80))
                    s.close()
                    
                    if result == 0:
                        print(f"Znalazłem urządzenie na {ip_to_check}")
                        ip_address = ip_to_check
                        self.settings['esp32_ip'] = ip_address
                        break
        except Exception as e:
            print(f"Błąd podczas skanowania sieci: {e}")
        
        # Próba połączenia z ESP32-CAM
        print(f"Próba połączenia z ESP32-CAM: {ip_address}")
        
        # Nie określamy z góry portu - metoda connect_to_ip_camera sama sprawdzi dostępne porty
        result = self.camera_manager.connect_to_ip_camera(ip_address)
        
        if result:
            # Ustaw flagę używania ESP32
            self.settings['use_esp32'] = True
            print(f"Połączono z ESP32-CAM: {ip_address}")
            
            # Zaktualizuj UI
            self.main_screen.connection_label.text = f"ESP32: {ip_address}"
            self.main_screen.connection_label.color = (0, 1, 0, 1)
        else:
            print(f"Nie udało się połączyć z ESP32-CAM: {ip_address}")
        
        return result
    
    def process_frame(self, dt):
        """Process frames from the camera"""
        if not self.is_capturing:
            return
        
        # Get frame from camera
        frame = self.camera_manager.get_frame()
        if frame is None:
            return
        
        # Make a copy for processing
        processed_frame = frame.copy()
        
        # Process face detection
        if self.settings['enable_face_detection'] and self.face_detector:
            try:
                faces_data = self.face_detector.process(processed_frame)
                for face_id, face_info in faces_data.items():
                    if face_info.get('alert'):
                        self.add_alert(f"Face {face_id} present for more than {self.settings['time_limit']} seconds")
            except Exception as e:
                print(f"Face detection error: {e}")
        
        # Process object detection
        if self.settings['enable_object_detection'] and self.object_detector:
            try:
                objects_data = self.object_detector.process(processed_frame)
                if objects_data['dangerous_objects']:
                    self.add_alert(f"Dangerous objects detected: {', '.join(objects_data['dangerous_objects'])}")
            except Exception as e:
                print(f"Object detection error: {e}")
        
        # Dodaj informację o ESP32-CAM jeśli używana
        if self.camera_manager.is_esp32_camera and self.debug_mode:
            cv2.putText(
                processed_frame, 
                f"ESP32-CAM: {self.settings['esp32_ip']}",
                (10, processed_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        # Update the UI with processed frame
        self.update_ui_with_frame(processed_frame)
    
    def update_ui_with_frame(self, frame):
        """Update the UI with the processed frame"""
        try:
            # Convert to texture for Kivy
            buf = cv2.flip(frame, 0)
            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            
            # Update the image widget
            self.main_screen.camera_view.texture = texture
        except Exception as e:
            print(f"Error updating UI with frame: {e}")
    
    def add_alert(self, message):
        """Add a new alert"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Check if this alert already exists (to avoid duplicates)
        for alert in self.alerts:
            if alert['message'] == message and time.time() - alert['raw_time'] < 5:
                return
        
        self.alerts.append({
            'message': message,
            'timestamp': timestamp,
            'raw_time': time.time(),
            'processed': False
        })
        
        # Update alerts screen if it's active
        if self.sm.current == 'alerts':
            self.alerts_screen.update_alerts(self.alerts)
        
        # Show alert indicator on main screen
        self.main_screen.show_alert_indicator()
        
        # Print alert message to console for debugging
        print(f"ALERT [{timestamp}]: {message}")
    
    def toggle_debug_mode(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        
        if self.object_detector:
            self.object_detector.set_debug_mode(self.debug_mode)
        
        if self.face_detector:
            self.face_detector.set_debug_mode(self.debug_mode)
            
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
    
    def on_stop(self):
        """Clean up resources when app is closed"""
        print("Zatrzymywanie aplikacji...")
        self.stop_camera()
        if self.camera_manager:
            self.camera_manager.release()
        print("Aplikacja zatrzymana")


if __name__ == "__main__":
    try:
        SecurityApp().run()
    except Exception as e:
        print(f"Błąd krytyczny: {e}")