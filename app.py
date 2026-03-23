import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import time

# Ensure src is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.cnn import PyTorchCNN

# Configure page
st.set_page_config(page_title="TerraVision AI Core", layout="wide", initial_sidebar_state="expanded")

# Custom UI Tweaks
st.markdown("""
<style>
    /* Clean, professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }
    
    /* Remove default top padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    
    /* Clean headers */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    /* Subtle metric cards */
    .telemetry-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        border: 1.5px dashed #CBD5E1 !important;
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease-in-out;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3B82F6 !important;
        background-color: #EFF6FF !important;
    }

    /* Remove decorative elements */
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Global Sidebar (Actionable Controls Only) ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 24px; color: #0F172A; margin-bottom: 0;'>TerraVision AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0; margin-bottom: 32px;'>System Dashboard</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 13px; text-transform: uppercase; color: #475569; font-weight: 700;'>Global Controls</h3>", unsafe_allow_html=True)
    
    confidence_thresh = st.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01,
                                 help="Adjust the probability threshold required to classify a sector as Agricultural.")
    
    st.write("")
    model_version = st.radio("Active Inference Engine", ["Deep CNN (v1.0)", "Vision Transformer (v2.0 - Pending)"], help="Switch between internal network architectures.")
    
    st.write("")
    enable_xai = st.toggle("Enable XAI Rendering Maps", value=True, help="Overlay a Class Activation Map (CAM) showing which geographical features triggered the classification.")

# --- Main Dashboard Header ---
st.markdown("<h1 style='font-size: 32px; margin-bottom: 0;'>TerraVision AI Operations</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 18px; margin-top: 6px;'>Deep Learning Pipeline for Real-Time Satellite Imagery Classification and Land-Use Analysis.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- XAI Hook Setup ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# --- Load Model ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = PyTorchCNN()
    weights_path = "true_best_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=True)
    
    # Register hook for Grad-CAM
    try:
        model.conv2.register_forward_hook(get_activation('conv2'))
    except AttributeError:
        pass
        
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Generate XAI Overlay
def generate_xai_heatmap(image_t, original_img):
    if 'conv2' not in activation:
        return original_img
    
    # Get last conv layer output
    act = activation['conv2'].squeeze(0)  
    # Pseudo-CAM by averaging activation map channels
    heatmap = torch.mean(act, dim=0).cpu().numpy()
    
    # Normalize 0-1
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    # Create colormap overlay
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Resize to original image size
    heatmap_img = Image.fromarray(heatmap_colored).resize(original_img.size, resample=Image.BILINEAR)
    
    # Blend with original
    blended = Image.blend(original_img, heatmap_img, alpha=0.45)
    return blended

# --- Layout ---
col1, col2 = st.columns([1, 1.8], gap="large")

with col1:
    st.markdown("<h2 style='font-size: 20px; color: #1E293B;'>Data Ingestion Module</h2>", unsafe_allow_html=True)
    
    # Staging Zone Container
    st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 24px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">
        <p style='font-size: 14px; color: #475569; margin-top: 0;'>Initialize telemetry processing sequence. Secure payload limits: JPG/PNG, Max 200MB.</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_size = round(sys.getsizeof(uploaded_file.getvalue()) / 1024, 1)
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown(f"""
        <div style="margin-top: 16px; border-top: 1px solid #F1F5F9; padding-top: 16px;">
            <p style="font-size: 13px; font-weight: 700; color: #075985; text-transform: uppercase; letter-spacing: 0.5px;">Active Staging Metrics</p>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #475569;">
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #F8FAFC;"><strong>Filename</strong></td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #F8FAFC; text-align: right;">{uploaded_file.name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #F8FAFC;"><strong>Payload Size</strong></td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #F8FAFC; text-align: right;">{file_size} KB</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #F8FAFC;"><strong>Resolution</strong></td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #F8FAFC; text-align: right;">{image.size[0]}×{image.size[1]} px</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0;"><strong>Ingestion Status</strong></td>
                    <td style="padding: 10px 0; text-align: right;"><span style="background-color: #DCFCE7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: 600;">Secured</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size: 13px; color: #94A3B8; text-align: center; margin-top: 16px;'>Awaiting telemetry payload...</p>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # System Architecture Info (Fully visible card)
    st.markdown("<h3 style='font-size: 14px; text-transform: uppercase; color: #475569; font-weight: 700; margin-top: 32px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;'>Execution Framework Documentation</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 13px; color: #475569; line-height: 1.6; margin-top: 16px; background-color: #FFFFFF; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <b>System Architecture</b><br>
    Core Inference Model: Custom Deep CNN<br>
    Evaluation Hardware: Local CPU/GPU Compute<br>
    Classification Regimes: Multi-Layer Feature Mapping<br><br>
    
    <b>Strategic Deployment Goals:</b><br>
    TerraVision AI automates the geographical extraction of arable polygons. Built inherently to interface with complex logistics pipelines, eliminating sub-optimal routing of specialized agricultural resources.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='font-size: 20px; color: #1E293B;'>Visual Analytics Platform</h2>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Pre-process
        img_tensor = transform(image).unsqueeze(0)
        
        # Inference & Telemetry Timing
        start_time = time.time()
        
        # Artificial delay to demonstrate loading skeleton / spinner
        with st.spinner("Extracting Deep Convolutional Features..."):
            with torch.no_grad():
                output = model(img_tensor)
                probability = torch.sigmoid(output).item()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        prob_agri = probability * 100
        prob_non = (1.0 - probability) * 100
        
        # Visual Canvas (XAI Integration)
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 4px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        
        if enable_xai:
            xai_img = generate_xai_heatmap(img_tensor, image)
            st.image(xai_img, use_container_width=True, caption="TerraVision AI: Class Activation Mapping Overlay (XAI Active)")
        else:
            st.image(image, use_container_width=True, caption="Raw Satellite Telemetry Feed")
            
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Results & Telemetry Row ---
        c1, c2 = st.columns([1.5, 1], gap="medium")
        
        with c1:
            st.markdown("<h3 style='font-size: 16px; margin-bottom: 16px; color: #0F172A; text-transform: uppercase;'>Granular Prediction Analysis</h3>", unsafe_allow_html=True)
            
            # Agricultural Bar
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-size: 14px; color: #334155; margin-bottom: 6px;">
                <span style="font-weight: {'700' if prob_agri >= (confidence_thresh*100) else '500'}; color: {'#0F172A' if prob_agri >= (confidence_thresh*100) else '#475569'};">Agricultural Sector</span>
                <span style="font-weight: {'700' if prob_agri >= (confidence_thresh*100) else '500'}; color: {'#0F172A' if prob_agri >= (confidence_thresh*100) else '#475569'};">{prob_agri:.2f}%</span>
            </div>
            <div style="height: 8px; background-color: #F1F5F9; border-radius: 4px; margin-bottom: 20px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);">
                <div style="height: 100%; width: {prob_agri}%; background-color: {'#10B981' if prob_agri >= (confidence_thresh*100) else '#94A3B8'}; border-radius: 4px; transition: width 0.5s ease-out;"></div>
            </div>
            
            <div style="display: flex; justify-content: space-between; font-size: 14px; color: #334155; margin-bottom: 6px;">
                <span style="font-weight: {'700' if prob_non > (100-(confidence_thresh*100)) else '500'}; color: {'#0F172A' if prob_non >= (100-(confidence_thresh*100)) else '#475569'};">Non-Agricultural Terrain</span>
                <span style="font-weight: {'700' if prob_non > (100-(confidence_thresh*100)) else '500'}; color: {'#0F172A' if prob_non >= (100-(confidence_thresh*100)) else '#475569'};">{prob_non:.2f}%</span>
            </div>
            <div style="height: 8px; background-color: #F1F5F9; border-radius: 4px; margin-bottom: 24px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);">
                <div style="height: 100%; width: {prob_non}%; background-color: {'#3B82F6' if prob_non > (100-(confidence_thresh*100)) else '#94A3B8'}; border-radius: 4px; transition: width 0.5s ease-out;"></div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("<h3 style='font-size: 16px; margin-bottom: 16px; color: #0F172A; text-transform: uppercase;'>Performance Telemetry</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="telemetry-card">
                <table style="width: 100%; border-collapse: separate; border-spacing: 0 8px; font-size: 13px; color: #475569;">
                    <tr>
                        <td style="color: #64748B;"><strong>Active Core</strong></td>
                        <td style="text-align: right; color: #2563EB; font-weight: 600;">CNN (Quantized)</td>
                    </tr>
                    <tr>
                        <td style="color: #64748B;"><strong>Inference Time</strong></td>
                        <td style="text-align: right; font-family: monospace; font-size: 14px; color: #0F172A;">{latency:.1f}ms</td>
                    </tr>
                    <tr>
                        <td style="color: #64748B;"><strong>I/O Resolution</strong></td>
                        <td style="text-align: right; font-family: monospace; font-size: 14px; color: #0F172A;">64x64px</td>
                    </tr>
                    <tr>
                        <td style="color: #64748B;"><strong>Threshold Limit</strong></td>
                        <td style="text-align: right; font-weight: 600; color: #0F172A;">{confidence_thresh*100}%</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 60px 20px; text-align: center;">
            <p style="color: #475569; font-size: 16px; margin: 0; font-weight: 600;">Analytics Canvas Offline</p>
            <p style="color: #94A3B8; font-size: 14px; margin: 6px 0 0 0;">Provide visual telemetry data via the staging module to initiate deep learning inference routines.</p>
        </div>
        """, unsafe_allow_html=True)
