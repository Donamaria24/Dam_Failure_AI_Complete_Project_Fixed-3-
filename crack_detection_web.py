# crack_detection_high_confidence.py
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import threading
import time

class HighConfidenceCrackDetection:
    def __init__(self):
        """Initialize with high confidence web alerts"""
        
        # Use YOUR actual trained model
        model_path = r"runs\detect\crack_detection4\weights\best.pt"
        
        print(f"🔍 Loading YOUR trained model: {model_path}")
        self.model = YOLO(model_path)
        
        # Print model classes
        print(f"✅ Model classes: {self.model.names}")
        
        # Find crack class IDs
        self.crack_class_ids = []
        for class_id, class_name in self.model.names.items():
            if any(word in class_name.lower() for word in ['crack', 'damage', 'defect', 'fracture']):
                self.crack_class_ids.append(class_id)
                print(f"✅ Crack class: '{class_name}' (ID: {class_id})")
        
        # Different thresholds for visual detection vs web alerts
        self.visual_confidence_threshold = 0.5    # Show detections ≥ 0.5 on screen
        self.web_alert_threshold = 0.7           # Send web alerts only ≥ 0.7 🎯
        
        self.web_server_url = "http://localhost:5000"
        
        print(f"\n🎯 CONFIDENCE SETTINGS:")
        print(f"   📺 Visual detection threshold: {self.visual_confidence_threshold}")
        print(f"   🌐 Web alert threshold: {self.web_alert_threshold}")
    
    def detect_cracks_with_thresholds(self, frame):
        """Detect cracks with different thresholds for visual vs web alerts"""
        
        # Run detection with lower threshold to catch more for visual display
        results = self.model(frame, conf=self.visual_confidence_threshold, verbose=False)
        
        visual_crack_found = False
        web_alert_crack_found = False
        max_confidence = 0
        web_alert_confidence = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    class_name = self.model.names.get(class_id, 'unknown')
                    
                    # Check if this is a crack class
                    if class_id in self.crack_class_ids:
                        
                        # Update max confidence for visual display
                        if confidence > max_confidence:
                            max_confidence = confidence
                        
                        # Visual detection (≥ 0.5)
                        if confidence >= self.visual_confidence_threshold:
                            visual_crack_found = True
                            print(f"📺 Visual crack: '{class_name}' ({confidence:.3f})")
                        
                        # Web alert detection (≥ 0.7) 🎯
                        if confidence >= self.web_alert_threshold:
                            web_alert_crack_found = True
                            web_alert_confidence = max(web_alert_confidence, confidence)
                            print(f"🌐 HIGH CONFIDENCE crack for web alert: '{class_name}' ({confidence:.3f})")
        
        return visual_crack_found, web_alert_crack_found, results, max_confidence, web_alert_confidence
    
    def send_high_confidence_alert(self, confidence):
        """Send web alert for high confidence cracks only"""
        def send_async():
            try:
                alert_data = {
                    'confidence': float(confidence),
                    'location': f'Mullaperiyar Dam - HIGH CONFIDENCE ({confidence:.2f})'
                }
                
                print(f"📨 Sending HIGH CONFIDENCE web alert: {confidence:.3f}")
                
                response = requests.post(
                    f"{self.web_server_url}/api/crack-detected",
                    json=alert_data,
                    timeout=3
                )
                
                if response.status_code == 200:
                    print(f"🌐 ✅ HIGH CONFIDENCE alert sent successfully! ({confidence:.3f})")
                else:
                    print(f"🌐 ❌ Web alert failed: {response.status_code}")
                    
            except Exception as e:
                print(f"🌐 ⚠️ Web alert error: {e}")
        
        # Send in background thread to avoid blocking
        threading.Thread(target=send_async, daemon=True).start()
    
    def process_camera_high_confidence(self):
        """Process camera with high confidence web alerts"""
        phone_ip = "192.168.1.8:8080"  # UPDATE WITH YOUR PHONE'S IP
        url = f"http://{phone_ip}/video"
        
        print(f"\n🎯 HIGH CONFIDENCE CRACK DETECTION")
        print(f"🔗 Connecting to: {url}")
        print(f"📺 Visual detection: Shows cracks ≥ {self.visual_confidence_threshold}")
        print(f"🌐 Web alerts: Only for cracks ≥ {self.web_alert_threshold}")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Camera connection failed!")
            return
        
        print("✅ Connected! High confidence mode active\n")
        
        last_web_alert_time = 0
        web_alert_cooldown = 5  # Only send web alerts every 5 seconds max
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect with different thresholds
            (visual_crack, web_crack, results, 
             max_conf, web_conf) = self.detect_cracks_with_thresholds(frame)
            
            # Draw visual results (all detections ≥ 0.5)
            if visual_crack and results:
                try:
                    annotated_frame = results[0].plot()
                except:
                    annotated_frame = frame
            else:
                annotated_frame = frame
            
            # Send web alert ONLY for high confidence cracks (≥ 0.7) 🎯
            if web_crack:
                current_time = time.time()
                if current_time - last_web_alert_time > web_alert_cooldown:
                    print(f"🚨 HIGH CONFIDENCE CRACK! Sending web alert ({web_conf:.3f})")
                    self.send_high_confidence_alert(web_conf)
                    last_web_alert_time = current_time
                else:
                    remaining = web_alert_cooldown - (current_time - last_web_alert_time)
                    print(f"⏳ High confidence crack detected but web alert on cooldown ({remaining:.1f}s)")
            
            # Status display
            if visual_crack:
                if web_crack:
                    status_text = f"🚨 HIGH CONF: {max_conf:.2f}"
                    color = (0, 0, 255)  # Red for high confidence
                else:
                    status_text = f"⚠️ LOW CONF: {max_conf:.2f}"
                    color = (0, 165, 255)  # Orange for low confidence
            else:
                status_text = "✅ No Crack"
                color = (0, 255, 0)  # Green
            
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.putText(annotated_frame, f"Visual: ≥{self.visual_confidence_threshold} | Web: ≥{self.web_alert_threshold}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(annotated_frame, "HIGH CONFIDENCE WEB ALERTS", 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Dam Crack Detection - HIGH CONFIDENCE ALERTS', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n🔚 High confidence detection stopped")

if __name__ == "__main__":
    print("🎯 Dam Failure AI - HIGH CONFIDENCE WEB ALERTS")
    print("🌐 Web alerts sent ONLY for cracks with confidence ≥ 0.7")
    print("📺 Visual detection shows all cracks ≥ 0.5")
    print("🔧 Make sure web server is running at http://localhost:5000\n")
    
    detector = HighConfidenceCrackDetection()
    detector.process_camera_high_confidence()
