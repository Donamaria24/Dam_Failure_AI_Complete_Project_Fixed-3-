# check_model.py
from ultralytics import YOLO
import os

def check_trained_model():
    """Check what classes are in your trained model"""
    
    # Check if your trained model exists
    model_paths = [
        'runs/detect/crack_detection/weights/best.pt',
        '../runs/detect/crack_detection/weights/best.pt',
        'best.pt'
    ]
    
    trained_model_found = False
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found trained model at: {path}")
            
            try:
                # Load YOUR trained model
                trained_model = YOLO(path)
                print(f"🔍 YOUR TRAINED MODEL classes: {trained_model.names}")
                
                # Check if it has crack-related classes
                crack_classes = []
                for class_id, class_name in trained_model.names.items():
                    if any(word in class_name.lower() for word in ['crack', 'damage', 'defect', 'fracture']):
                        crack_classes.append(f"'{class_name}' (ID: {class_id})")
                
                if crack_classes:
                    print(f"✅ Found crack classes: {crack_classes}")
                else:
                    print("❌ NO CRACK CLASSES FOUND in your trained model!")
                
                trained_model_found = True
                break
                
            except Exception as e:
                print(f"❌ Error loading model from {path}: {e}")
    
    if not trained_model_found:
        print("❌ NO TRAINED MODEL FOUND!")
        print("This means your training didn't complete or files are in wrong location")
    
    # For comparison, show what pre-trained model has
    print(f"\n📊 PRE-TRAINED MODEL classes (for comparison):")
    pretrained_model = YOLO('yolov8n.pt')
    print(f"Pre-trained classes: {pretrained_model.names}")
    print(f"Pre-trained has {len(pretrained_model.names)} classes (person, laptop, keyboard, etc.)")

if __name__ == "__main__":
    check_trained_model()
