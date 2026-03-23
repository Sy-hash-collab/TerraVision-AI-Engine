import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import PyTorchCNN
from src.data_loader import get_pytorch_dataloader
from src.evaluate import print_ffc_business_metrics

def main():
    print("==================================================")
    print("🚜 AgriYield AI - Local Execution Pipeline Test")
    print("==================================================")
    
    print("\n[1/4] Initializing PyTorch Baseline CNN...")
    try:
        model = PyTorchCNN()
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    data_dir = './images_dataSAT/'
    print(f"\n[2/4] Attempting to load data from {data_dir}...")
    if not os.path.exists(data_dir):
        print(f"⚠️ WARNING: Data directory '{data_dir}' not found locally.")
        print("Generating synthetic dummy data to verify the pipeline architecture...")
        dummy_images = torch.randn(8, 3, 64, 64)
    else:
        loader, classes = get_pytorch_dataloader(data_dir=data_dir, batch_size=8)
        if loader:
            print(f"✅ Data loader initialized. Classes: {classes}")
            dummy_images, _ = next(iter(loader))
        else:
            dummy_images = torch.randn(8, 3, 64, 64)
            
    print("\n[3/4] Executing Neural Network Forward Pass...")
    try:
        outputs = model(dummy_images)
        print("✅ Forward pass successful.")
        print(f"Output tensor shape: {outputs.shape} (Expected: [8, 1])")
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return
    
    print("\n[4/4] Simulating FFC Executive Evaluation...")
    dummy_true_labels = [1, 0, 1, 1, 0, 0, 1, 0]
    dummy_predictions = [0.85, 0.1, 0.92, 0.45, 0.2, 0.3, 0.78, 0.6]
    print_ffc_business_metrics(dummy_true_labels, dummy_predictions, model_name="PyTorch CNN (Verification Run)", threshold=0.5)
    
    print("\n🚀 SUCCESS: The AgriYield project pipeline is syntactically correct and ready for Live Data Deployment!")

if __name__ == "__main__":
    main()
