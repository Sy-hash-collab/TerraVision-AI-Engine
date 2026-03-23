import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import PyTorchCNN
from src.data_loader import get_pytorch_dataloader
from src.evaluate import print_ffc_business_metrics

def main():
    print("==================================================")
    print("🚜 AgriYield AI - FULL TRAINING & EVALUATION PIPELINE")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # 1. Load Data
    data_dir = './images_dataSAT/'
    batch_size = 32
    print(f"[*] Loading dataset from {data_dir} with batch_size={batch_size}...")
    loader, classes = get_pytorch_dataloader(data_dir=data_dir, batch_size=batch_size)
    
    if loader is None:
        print("[!] Failed to load data. Exiting.")
        return
        
    print(f"[*] Detected Classes: {classes}")
    
    # 2. Init Model
    print("[*] Initializing PyTorch CNN...")
    model = PyTorchCNN().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 1  
    print(f"[*] Starting Training for {epochs} epoch(s)...")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # Limit to 20 batches to demonstrate functionality quickly without making you wait hours
        max_batches = 20
        batch_count = 0
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if batch_count >= max_batches:
                break
                
        print(f"    Loss after {max_batches} batches: {running_loss / max_batches:.4f}")
        
    print("[*] Training Complete. Saving custom architecture weights...")
    torch.save(model.state_dict(), 'true_best_model.pth')
    
    # 4. Evaluation
    print("\n[*] Running Evaluation Phase...")
    model.eval()
    all_preds = []
    all_labels = []
    
    eval_batches = 10
    eval_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            eval_count += 1
            if eval_count >= eval_batches:
                break
                
    # 5. Print Metrics
    print("\n" + "="*50)
    print_ffc_business_metrics(all_labels, all_preds, model_name="PyTorch CNN (Quick Training Run)", threshold=0.5)
    print("==================================================")
    print("✅ Pipeline successfully trained and evaluated on real data!")

if __name__ == "__main__":
    main()
