# 🚜 AgriYield AI: Arable Land Mapping for Strategic Fertilizer Distribution (FFC)

This project leverages Keras and PyTorch to classify agricultural land from satellite imagery, utilizing both lightweight CNNs and SOTA Vision Transformers. 

## 📂 Project Structure
* `data/` or `images_dataSAT/` - The satellite dataset required for training.
* `src/data_loader.py` - Contains custom data generators and pipelines to handle augmented data efficiently.
* `src/models/cnn.py` - Contains Keras and PyTorch Sequential CNN architectures.
* `src/models/vit.py` - Contains the advanced Hybrid CNN-Vision Transformers.
* `src/evaluate.py` - Contains specialized metrics (F1-Score, Confusion Matrices) mapped to FFC's business logic.
* `src/main.py` - The integration script that ties the pipeline together for local execution.

## 🚀 How to Run the Project Locally

### Step 1: Open Your Terminal
Open PowerShell or your VSCode/JupyterLab Terminal and navigate to the Capstone directory:
```powershell
cd "c:\Users\SMART\OneDrive\Desktop\Computer Science\Coursera\IBM AI Engineering\C6 AI CApstone Project"
```

### Step 2: Verify Your Dataset
Ensure the `images_dataSAT/` folder is present in the root directory. It should contain two subfolders:
- `class_0_non_agri`
- `class_1_agri`

### Step 3: Run the Integration Verification Pipeline
To ensure your python environment has all the required dependencies (TensorFlow, PyTorch, scikit-learn) and to test that the model wires correctly to the data, run:
```powershell
python src\main.py
```
*If this executes cleanly, your environment is perfect.*

### Step 4: Complete Your Jupyter Notebooks
Now that your core ML engineering is modularized in `src/`, open your Capstone Jupyter Notebooks (e.g., `Lab-M2L2...ipynb`). 
You can import your models directly into the Jupyter cells instead of writing massive code blocks from scratch!

Example for PyTorch notebook:
```python
# Instead of writing the whole network architecture manually:
from src.models.cnn import PyTorchCNN
from src.data_loader import get_pytorch_dataloader

model = PyTorchCNN()
train_loader, classes = get_pytorch_dataloader('./images_dataSAT/')
```

### Step 5: Generate The FFC Report Notebook
In your final Module 4 notebook, import the specialized business metrics script:
```python
from src.evaluate import print_ffc_business_metrics

# Use this to print the final verdict!
print_ffc_business_metrics(y_true, y_pred, model_name="PyTorch CNN-ViT Hybrid")
```
