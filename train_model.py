import torch
from ultralytics import YOLO

def check_gpu_availability():
    """Check if CUDA and GPU are available"""
    print("=== GPU Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ CUDA not available. Training will use CPU (much slower)")
        return False

def train():
    """Train YOLOv8 model with GPU acceleration"""
    
    # Check GPU availability first
    gpu_available = check_gpu_availability()
    
    # Load YOLOv8 nano model (optimized for your 8GB VRAM)
    model = YOLO('yolov8n.pt')
    
    # GPU-optimized training parameters
    training_params = {
        'data': 'https://github.com/ultralytics/ultralytics/raw/main/ultralytics/cfg/datasets/crack-seg.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,  # Optimal batch size for RTX 4060 8GB
        'name': 'crack_detection',
        'patience': 20,
        'save': True,
        'plots': True,
        'device': 0 if gpu_available else 'cpu',  # Use GPU 0 if available
        'amp': True,  # Mixed precision training - saves VRAM and speeds up training
        'cache': True,  # Cache images for faster training
        'workers': 8,  # Multi-threading for data loading
    }
    
    print("\n=== Starting Training ===")
    print(f"Device: {'GPU (CUDA)' if gpu_available else 'CPU'}")
    print(f"Batch size: {training_params['batch']}")
    print(f"Image size: {training_params['imgsz']}")
    print(f"Mixed precision: {training_params['amp']}")
    
    try:
        # Start training with GPU acceleration
        results = model.train(**training_params)
        
        print("\n✅ Training completed successfully!")
        print(f"Model saved to: runs/detect/crack_detection/weights/best.pt")
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU Out of Memory!")
            print("Trying with reduced batch size...")
            
            # Retry with smaller batch size
            training_params['batch'] = 8
            training_params['amp'] = True  # Ensure mixed precision is enabled
            
            try:
                results = model.train(**training_params)
                print("\n✅ Training completed with reduced batch size!")
                return results
            except Exception as e2:
                print(f"❌ Training failed even with reduced settings: {e2}")
                return None
        else:
            print(f"❌ Training error: {e}")
            return None

if __name__ == '__main__':
    train()
