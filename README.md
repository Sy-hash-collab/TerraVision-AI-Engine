# TerraVision AI: Geospatial Classification Engine

**TerraVision AI** is a production-grade Deep Learning pipeline engineered for the real-time classification of high-resolution satellite imagery. It automates topographical analysis to optimize supply chain logistics, geographical assessments, and automated resource allocation for agricultural enterprises.

This repository demonstrates the transition from exploratory data science to deployed Software Engineering, encapsulating advanced Neural Architectures within a modular, Object-Oriented framework.

---

## Enterprise Architecture Overview

The codebase is structured for strict separation of concerns, ensuring maximum maintainability, scalability, and reproducibility.

### Core Technical Specs
- **Model Architectures:** Custom Deep Convolutional Neural Networks (CNN) and experimental Vision Transformer (ViT) Hybrid models.
- **Explainable AI (XAI):** Integrated Class Activation Mapping (Grad-CAM) via custom PyTorch forward hooks. This visualizes exactly which tensor features trigger the network's final classification layer.
- **Inference Latency:** Sub-second predictive rendering (approx. 120ms compute overhead) via the deployed frontend.
- **Frameworks & Tooling:** PyTorch, Keras/TensorFlow, Streamlit, Scikit-Learn.
- **Domain Focus:** Computer Vision (CV), Geospatial Analysis, Supply Chain Optimization, MLOps.

---

## Repository Structure

- `src/data_loader.py`: Enterprise-grade data pipelines implementing iterative batch generation, randomized tensor augmentations (rotation, flip), and dynamic spatial resizing.
- `src/models/cnn.py`: Object-Oriented class definitions of the core PyTorch CNN architecture, built from generic `nn.Conv2d` blocks.
- `src/models/vit.py`: Hybrid CNN-Vision Transformer definitions aiming to capture global contextual dependencies across satellite feeds.
- `src/evaluate.py`: Custom evaluation functions translating raw confusion matrices, precision, and recall ratios into actionable business intelligence.
- `src/run_full_pipeline.py`: Automated orchestration script handling local training loops, backpropagation via Adam optimization, and `.pth` weight persistence.
- `app.py`: The production frontend. A highly polished, low-latency deployment dashboard built with Streamlit, rendering live inference metrics and XAI heatmaps.

---

## Quick Start (Local Execution)

### Environment Setup

Ensure you have Python 3.9+ installed natively or via Conda. Clone the repository and install the production dependencies:

```bash
git clone https://github.com/Sy-hash-collab/TerraVision-AI-Engine.git
cd TerraVision-AI-Engine

# Create a secure virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
# source venv/bin/activate

# Install Core ML Dependencies
pip install -r requirements.txt
```

### Initiating the Web Application

To interface with the model directly and utilize the visual XAI telemetry suite, start the deployment server:

```bash
streamlit run app.py
```
The application will bridge to your local environment (defaults to `http://localhost:8501`).

### Training Pipeline Execution

If you wish to replicate the training procedure against the raw satellite imagery corpus and generate fresh `.pth` weights recursively:

```bash
python src/run_full_pipeline.py
```
This script initializes the custom PyTorch networks, binds the iterative dataloaders, computes gradients across batches, logs categorical cross-entropy loss metrics, and saves the highly optimized tensor state dict.

---

## Engineering Philosophy

This pipeline proves the capability to move deep learning out of isolated Jupyter Notebooks and into robust software structures. 

By abstracting models as Python classes and defining unified interfaces for data ingestion, the system allows for rapid architecture swapping (e.g., standard CNN vs Hybrid ViT) without disrupting the downstream inference engine.

Furthermore, integrating real-time activation mapping (XAI) directly into the UI layer guarantees that AI predictions remain transparent, explainable, and accountable to human auditing.
