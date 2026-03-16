# find_model.py
import os
import glob

def find_all_pt_files():
    """Find all .pt files in your system"""
    
    print("🔍 Searching for all .pt files...")
    
    # Search in common locations
    search_paths = [
        "**/*.pt",
        "runs/**/*.pt", 
        "../**/*.pt",
        "../../**/*.pt"
    ]
    
    all_pt_files = []
    
    for pattern in search_paths:
        files = glob.glob(pattern, recursive=True)
        all_pt_files.extend(files)
    
    if all_pt_files:
        print(f"✅ Found {len(all_pt_files)} .pt files:")
        for i, file in enumerate(all_pt_files):
            size = os.path.getsize(file) / (1024*1024)  # Size in MB
            print(f"   {i+1}. {file} ({size:.1f} MB)")
        return all_pt_files
    else:
        print("❌ No .pt files found!")
        return []

if __name__ == "__main__":
    find_all_pt_files()
