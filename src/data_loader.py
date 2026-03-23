import os
import glob
import random
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_keras_paths(data_dir):
    """Answers Q2 Task 1 & 2: Create all_image_paths and bind logic"""
    all_image_paths = glob.glob(os.path.join(data_dir, '*', '*.jpg'))
    labels = [1 if 'class_1_agri' in path else 0 for path in all_image_paths]
    temp = list(zip(all_image_paths, labels))
    random.shuffle(temp)
    return temp

def custom_data_generator(data, batch_size=8):
    """Answers Q2 Task 3 & 4: Custom Generator logic (Stub for iterative yielding)"""
    while True:
        batch = random.sample(data, batch_size)
        # Add actual image loading and yielding logic here using cv2/keras
        yield batch

def get_pytorch_dataloader(data_dir='./images_dataSAT/', batch_size=8):
    """Answers Q3 Tasks 1-5: PyTorch Transforms and DataLoaders"""
    custom_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])
    # Note: Ensure the data_dir points to the parent directory containing the class folders
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found. Please ensure data is loaded.")
        return None, None
        
    dataset = datasets.ImageFolder(root=data_dir, transform=custom_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes

if __name__ == "__main__":
    print("AgriYield AI - Data Loader Component Initialized.")
