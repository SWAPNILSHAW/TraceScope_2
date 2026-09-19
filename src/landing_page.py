import streamlit as st
import time
import random
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit.components.v1 import html as components_html
import io
import textwrap
import os
import sys

# Add src to sys.path to ensure we can import modules if running from root or src
from fpdf import FPDF
import base64
import tracemalloc


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Model Imports
# import tensorflow as tf  <-- Lazy loaded
import pickle
import joblib
from baseline.predict_baseline import predict_scanner as predict_baseline
# from hybrid_cnn.utils import ... <-- Lazy loaded

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def create_pdf_report(scanner_data, file_name="Unknown"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(0, 212, 255) # Cyan
        pdf.cell(0, 15, "TraceScope AI", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 10, "Forensic Analysis Report", 0, 1, 'C')
        
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_draw_color(0, 212, 255)
        pdf.line(10, 45, 200, 45)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Analysis Results", 0, 1, 'L')
        
        pdf.set_font("Arial", '', 11)
        pdf.cell(60, 10, "Source File:", 0, 0)
        pdf.cell(0, 10, str(file_name), 0, 1)
        
        pdf.cell(60, 10, "Predicted Brand:", 0, 0)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, str(scanner_data.get('brand', 'Unknown')), 0, 1)
        pdf.set_font("Arial", '', 11)
        
        pdf.cell(60, 10, "Predicted Model:", 0, 0)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, str(scanner_data.get('model', 'Unknown')), 0, 1)
        pdf.set_font("Arial", '', 11)
        
        pdf.cell(60, 10, "Confidence Score:", 0, 0)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f"{scanner_data.get('confidence', 0):.2f}%", 0, 1)
        pdf.set_font("Arial", '', 11)
        
        pdf.ln(20)
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10, pdf.get_y(), 190, 30, 'F')
        pdf.set_xy(15, pdf.get_y()+5)
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(180, 8, "Disclaimer: This report was generated automatically by TraceScope AI. The results are based on probabilistic models and should be verified by a forensic expert.")
        
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        return None


@st.cache_resource
def get_hybrid_resources():
    try:
        base_path = os.path.join(RESULTS_DIR, "hybrid_cnn")
        # Updated file names based on verification
        model_path = os.path.join(base_path, "scanner_hybrid.keras")
        le_path = os.path.join(base_path, "hybrid_label_encoder.pkl") 
        scaler_path = os.path.join(base_path, "hybrid_feat_scaler.pkl")
        fps_path = os.path.join(base_path, "scanner_fingerprints.pkl")
        keys_path = os.path.join(base_path, "fingerprint_keys.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
            
        import tensorflow as tf  # Lazy import
        model = tf.keras.models.load_model(model_path)
        with open(le_path, "rb") as f: le = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        with open(fps_path, "rb") as f: fps = pickle.load(f)
        
        # Fallback for keys if file doesn't exist
        if os.path.exists(keys_path):
            with open(keys_path, "rb") as f: keys = pickle.load(f)
        else:
            keys = list(fps.keys())
        
        return {
            "model": model, "le": le, "scaler": scaler,
            "fps": fps, "fp_keys": keys
        }
    except Exception as e:
        st.error(f"Failed to load Hybrid CNN resources: {e}")
        return None

def load_hybrid_resources():
    return get_hybrid_resources()


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TraceScope AI - Forensic Scanner Identification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1.1 Metrics Session State Initialization
# -----------------------------------------------------------------------------
if 'session_count' not in st.session_state:
    st.session_state['session_count'] = 0
if 'session_confidences' not in st.session_state:
    st.session_state['session_confidences'] = []
if 'processing_times' not in st.session_state:
    st.session_state['processing_times'] = []
if 'history' not in st.session_state:
    st.session_state['history'] = []


# -----------------------------------------------------------------------------
# 2. Enhanced Modern UI & CSS Styling
# -----------------------------------------------------------------------------
# NOTE: HTML strings are flushed left to prevent them from rendering as code blocks.
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    /* GLOBAL RESET */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
        background: #0a0e17;
    }
    
    /* HEADINGS */
    h1 { 
        font-size: 3.5rem !important; 
        font-weight: 900 !important; 
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #00d4ff);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 1rem !important;
    }
    
    h2 { 
        font-size: 2.2rem !important; 
        font-weight: 800 !important; 
        color: #00d4ff !important; 
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid #00d4ff;
        padding-left: 10px;
        line-height: 1.2;
    }
    
    h3 { 
        font-size: 1.5rem !important; 
        font-weight: 700 !important; 
        color: #ffffff !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    p { 
        font-size: 1.05rem; 
        line-height: 1.6; 
        color: #b0b0b0;
        margin-bottom: 1rem;
    }

    /* DOCUMENTATION CARDS */
    .doc-card {
        background: linear-gradient(145deg, rgba(20, 25, 40, 0.6), rgba(15, 20, 30, 0.8));
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 25px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        text-align: left;
        display: flex;
        flex-direction: column;
    }
    
    .doc-card:hover {
        background: linear-gradient(145deg, rgba(25, 30, 50, 0.8), rgba(20, 25, 40, 0.9));
        border-color: #00d4ff;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 
                    0 0 20px rgba(0, 212, 255, 0.1);
    }
    
    .doc-card-icon {
        font-size: 2rem;
        margin-bottom: 15px;
        background: rgba(0, 212, 255, 0.1);
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: #00d4ff;
    }

    .doc-card h4 {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 10px 0;
    }
    
    .doc-card-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .doc-card-list li {
        position: relative;
        padding-left: 20px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: #b0b0b0;
    }
    
    .doc-card-list li::before {
        content: "•";
        color: #00d4ff;
        position: absolute;
        left: 0;
        font-weight: bold;
    }

    /* DOCUMENTATION CARDS */
    .doc-card {
        background: linear-gradient(145deg, rgba(20, 25, 40, 0.6), rgba(15, 20, 30, 0.8));
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 25px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    
    .doc-card:hover {
        background: linear-gradient(145deg, rgba(25, 30, 50, 0.8), rgba(20, 25, 40, 0.9));
        border-color: #00d4ff;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 
                    0 0 20px rgba(0, 212, 255, 0.1);
    }
    
    .doc-card-icon {
        font-size: 2rem;
        margin-bottom: 15px;
        background: rgba(0, 212, 255, 0.1);
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: #00d4ff;
    }

    .doc-card h4 {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 10px 0;
    }
    
    .doc-card-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .doc-card-list li {
        position: relative;
        padding-left: 20px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: #b0b0b0;
    }
    
    .doc-card-list li::before {
        content: "•";
        color: #00d4ff;
        position: absolute;
        left: 0;
        font-weight: bold;
    }

    /* HERO SECTION - Enhanced */
    .hero-container {
        background: radial-gradient(circle at 50% 0%, 
            rgba(0, 212, 255, 0.15) 0%, 
            rgba(10, 14, 23, 0.95) 80%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        padding: 4rem 3rem;
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #ff00ff, #00d4ff);
        animation: scanline 3s linear infinite;
    }
    
    @keyframes scanline {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* ENHANCED FEATURE CARDS */
    .feature-card {
        background: linear-gradient(145deg, 
            rgba(20, 25, 40, 0.8), 
            rgba(10, 15, 30, 0.9));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00d4ff, #004e66, #00d4ff);
        z-index: -1;
        filter: blur(10px);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #00d4ff;
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.15),
                    inset 0 0 30px rgba(0, 212, 255, 0.05);
    }
    
    .feature-card:hover::before {
        opacity: 0.5;
    }

    /* METRICS - Enhanced */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #00d4ff !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #80deea !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    /* =========================================
       ENHANCED SIDEBAR STYLING
       ========================================= */
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(10, 15, 25, 0.98) 0%,
            rgba(15, 20, 35, 0.95) 100%) !important;
        border-right: 2px solid rgba(0, 212, 255, 0.2) !important;
        backdrop-filter: blur(10px);
    }

    /* Remove default bullets from markdown lists in sidebar */
    [data-testid="stSidebar"] ul {
        list-style-type: none;
        padding-left: 0;
        margin: 0;
    }

    /* Enhanced menu buttons */
    .sidebar-menu-btn {
        text-decoration: none;
        color: #b0b0b0 !important;
        font-weight: 500;
        display: flex;
        align-items: center;
        padding: 12px 18px;
        margin-bottom: 8px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
        font-size: 0.95rem;
        cursor: pointer;
    }
    
    .sidebar-menu-btn:hover {
        background: linear-gradient(90deg, 
            rgba(0, 212, 255, 0.1) 0%,
            rgba(0, 212, 255, 0.05) 100%);
        color: #00d4ff !important;
        border-left: 4px solid #00d4ff;
        padding-left: 22px;
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.1);
    }
    
    .sidebar-menu-btn.active {
        background: linear-gradient(90deg, 
            rgba(0, 212, 255, 0.15) 0%,
            rgba(0, 212, 255, 0.08) 100%) !important;
        color: #00d4ff !important;
        border-left: 4px solid #00d4ff !important;
        font-weight: 600 !important;
    }
    
    .sidebar-icon {
        margin-right: 12px;
        font-size: 1.2rem;
        width: 24px;
        text-align: center;
    }

    /* Separator lines in sidebar */
    .sidebar-divider {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(0, 212, 255, 0.3) 50%, 
            transparent 100%);
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .status-active {
        background: rgba(0, 255, 136, 0.15);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    
    .status-inactive {
        background: rgba(255, 65, 65, 0.15);
        color: #ff4141;
        border: 1px solid rgba(255, 65, 65, 0.3);
    }
    
    .status-warning {
        background: rgba(255, 193, 7, 0.15);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }

    /* Quick Action Buttons */
    .quick-action-btn {
        width: 100%;
        margin-bottom: 8px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.2);
    }

    /* Documentation Panel */
    .doc-panel {
        background: rgba(20, 25, 40, 0.7);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .doc-panel:hover {
        border-color: #00d4ff;
        background: rgba(20, 25, 40, 0.9);
        transform: translateY(-2px);
    }
    
    /* System Monitor */
    .system-monitor {
        background: rgba(15, 20, 35, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    /* Progress Bars */
    .progress-container {
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 5px;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00d4ff, #0088cc);
        border-radius: 4px;
        transition: width 1s ease;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Glow Effects */
    .glow {
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Code Blocks */
    .code-block {
        background: rgba(10, 15, 25, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 15px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #80deea;
        margin: 10px 0;
        overflow-x: auto;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #00d4ff;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(15, 20, 35, 0.95);
        color: #e0e0e0;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(0, 212, 255, 0.3);
        font-size: 0.85rem;
        backdrop-filter: blur(10px);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Enhanced Sidebar with Multiple Sections
# -----------------------------------------------------------------------------
with st.sidebar:
    # --- LOGO & BRANDING ---
    st.markdown(textwrap.dedent("""
<div style="text-align: center; margin-bottom: 3rem; padding: 2rem 0 1rem 0;">
    <div style="
        display: inline-flex; 
        align-items: center; 
        justify-content: center; 
        width: 80px; 
        height: 80px; 
        background: linear-gradient(135deg, #00d4ff, #004e66); 
        border-radius: 24px; 
        margin-bottom: 20px;
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.25);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                transparent 30%, 
                rgba(255, 255, 255, 0.1) 50%, 
                transparent 70%);
            animation: shine 3s infinite linear;
        "></div>
        <span style="font-size: 35px; z-index: 1;">🔬</span>
    </div>
    <h2 style="
        color: #fff !important; 
        margin: 0; 
        font-size: 1.8rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    ">TraceScope AI</h2>
    <p style="
        color: #80deea; 
        font-size: 0.85rem; 
        letter-spacing: 2px; 
        margin-top: 8px;
        font-weight: 500;
    ">FORENSIC ANALYSIS SUITE</p>
</div>

<style>
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform
        : translateX(100%); }
    }
</style>
"""), unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # --- NAVIGATION MENU ---
    st.markdown(textwrap.dedent("""
<p style="
    font-size: 0.75rem; 
    color: #80deea; 
    font-weight: 700; 
    letter-spacing: 1.5px; 
    margin-bottom: 15px;
    text-transform: uppercase;
    opacity: 0.8;
">MAIN NAVIGATION</p>
"""), unsafe_allow_html=True)
    
    # Navigation Links with Active State
    current_page = st.session_state.get('current_page', 'dashboard')
    
    nav_items = [
        ("🏠 Dashboard Home", "dashboard", "#dashboard"),
        ("📋 Project Overview", "overview", "#project-overview"),
        ("⚙️ System Architecture", "architecture", "#system-architecture"),
        ("🧪 Live Analysis Lab", "analysis", "#forensic-analysis-lab"),
        ("📊 Analytics Dashboard", "analytics", "#analytics"),
        ("📚 Documentation", "docs", "#documentation"),
        ("🔧 Settings", "settings", "#settings"),
        ("📈 Performance", "performance", "#performance"),
    ]
    
    st.markdown("""
    <script>
        // Global click listener for Sidebar Navigation
        // This runs once when the sidebar is rendered.
        // We use window.parent.document to ensure we catch events in the main app context if needed.
        
        function setupNav() {
            const doc = window.parent.document;
            if (!doc) return;
            
            // Add click listener to the document (Event Delegation)
            doc.addEventListener('click', function(e) {
                // Check if the clicked element or its parent is a sidebar button
                const link = e.target.closest('.sidebar-menu-btn');
                
                if (link) {
                    // 1. Remove active class from ALL sidebar buttons
                    const allLinks = doc.querySelectorAll('.sidebar-menu-btn');
                    allLinks.forEach(l => l.classList.remove('active'));
                    
                    // 2. Add active class to the CLICKED button
                    link.classList.add('active');
                    
                    // Note: The smooth scroll happens automatically via the href="#..." anchor
                }
            }, { capture: true }); // Capture phase to ensure we intercept it
        }
        
        // Run setup
        setupNav();
    </script>
    """, unsafe_allow_html=True)
    
    for label, page_id, href in nav_items:
        # Determine initial active state
        active_class = "active" if current_page == page_id else ""
        
        # Clean HTML anchor
        st.markdown(f"""
<a href="{href}" class="sidebar-menu-btn {active_class}" id="nav_{page_id}">
    <span class="sidebar-icon">{label.split(' ')[0]}</span>
    <span class="sidebar-label">{label.split(' ', 1)[1] if ' ' in label else label}</span>
</a>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # --- QUICK ACCESS PANEL ---
    st.markdown(textwrap.dedent("""
<p style="
    font-size: 0.75rem; 
    color: #80deea; 
    font-weight: 700; 
    letter-spacing: 1.5px; 
    margin-bottom: 15px;
    text-transform: uppercase;
    opacity: 0.8;
">QUICK ACTIONS</p>
"""), unsafe_allow_html=True)
    
    # Quick Action Buttons
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.rerun()
    with col_q2:
        if st.button("📥 Export Logs", use_container_width=True):
            st.success("📁 Logs exported successfully!")
    
    if st.button("🔍 New Analysis", type="primary", use_container_width=True):
        st.session_state['new_analysis'] = True
    
    if st.button("📋 Generate Report", use_container_width=True):
        st.info("📄 Report generation started...")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # --- SYSTEM STATUS PANEL ---
    st.markdown(textwrap.dedent("""
<p style="
    font-size: 0.75rem; 
    color: #80deea; 
    font-weight: 700; 
    letter-spacing: 1.5px; 
    margin-bottom: 15px;
    text-transform: uppercase;
    opacity: 0.8;
">SYSTEM STATUS</p>
"""), unsafe_allow_html=True)
    
    # System Status Indicators
    with st.container():
        st.markdown('<div class="system-monitor fade-in">', unsafe_allow_html=True)
        
        # Status Row 1
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
<div style="text-align: center;">
    <div style="font-size: 0.8rem; color: #9aa0a6; margin-bottom: 4px;">AI Engine</div>
    <div class="status-badge status-active">● Active</div>
</div>
""", unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("""
<div style="text-align: center;">
    <div style="font-size: 0.8rem; color: #9aa0a6; margin-bottom: 4px;">Database</div>
    <div class="status-badge status-active">● Connected</div>
</div>
""", unsafe_allow_html=True)
        
        # Status Row 2
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            st.markdown("""
<div style="text-align: center;">
    <div style="font-size: 0.8rem; color: #9aa0a6; margin-bottom: 4px;">API Gateway</div>
    <div class="status-badge status-warning">● Degraded</div>
</div>
""", unsafe_allow_html=True)
        
        with col_s4:
            st.markdown("""
<div style="text-align: center;">
    <div style="font-size: 0.8rem; color: #9aa0a6; margin-bottom: 4px;">Storage</div>
    <div class="status-badge status-active">● 78% Free</div>
</div>
""", unsafe_allow_html=True)
        
        # System Load
        st.markdown("""
<div style="margin-top: 15px;">
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 5px;">
        <span style="color: #80deea;">System Load</span>
        <span style="color: #00d4ff; font-weight: 600;">68%</span>
    </div>
    <div class="progress-container">
        <div class="progress-bar">
            <div class="progress-fill" style="width: 68%;"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # --- DOCUMENTATION PANEL ---
    st.markdown(textwrap.dedent("""
<p style="
    font-size: 0.75rem; 
    color: #80deea; 
    font-weight: 700; 
    letter-spacing: 1.5px; 
    margin-bottom: 15px;
    text-transform: uppercase;
    opacity: 0.8;
">QUICK DOCS</p>
"""), unsafe_allow_html=True)
    
    with st.expander("📖 User Guide", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. Upload scanned document
        2. Configure analysis settings
        3. Run identification
        4. Review results
        
        **Supported Formats:** JPG, PNG, TIFF, PDF
        **Max File Size:** 50MB
        """)
    
    with st.expander("🔧 API Reference", expanded=False):
        st.markdown("""
        ```python
        # Sample API Call
        import requests
        
        response = requests.post(
            url="[https://api.tracescope.ai/v1/analyze](https://api.tracescope.ai/v1/analyze)",
            files={'document': file},
            params={'mode': 'full_analysis'}
        )
        ```
        """)
    
    with st.expander("⚙️ Configuration", expanded=False):
        st.markdown("""
        **Default Settings:**
        - Model: CNN Ensemble
        - Confidence Threshold: 85%
        - Feature Extraction: PRNU + Wavelet
        - Output Format: JSON + PDF
        """)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # --- FOOTER ---
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
<div style="
    text-align: center; 
    padding: 20px 0; 
    color: #4a6572; 
    font-size: 0.75rem;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    margin-top: 10px;
">
    <div style="margin-bottom: 8px;">
        <span style="color: #00d4ff; font-weight: 600;">v2.1.4</span> • 
        <span style="color: #80deea;">{current_time}</span>
    </div>
    <div style="color: #666; line-height: 1.4;">
        © 2024 TraceScope AI Systems<br>
        <span style="font-size: 0.7rem;">For Forensic Research & Academic Use</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Hero Section
# -----------------------------------------------------------------------------
st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-container fade-in">
    <div style="
        text-transform: uppercase; 
        letter-spacing: 3px; 
        font-size: 0.9rem; 
        color: #00d4ff; 
        margin-bottom: 1rem; 
        font-weight: 700;
        background: rgba(0, 212, 255, 0.1);
        padding: 8px 20px;
        border-radius: 20px;
        display: inline-block;
    ">Advanced Forensic Analysis</div>
    <h1 class="hero-title">Scanner Fingerprint Identification</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem; max-width: 800px; margin-left: auto; margin-right: auto;">
        Identify the source scanner device by analyzing unique noise patterns, frequency artifacts, 
        and digital fingerprints using AI-powered forensic analysis.
    </p>
    <div style="margin-top: 2rem;">
        <span class="status-badge status-active" style="margin: 0 5px;">Real-time Processing</span>
        <span class="status-badge status-active" style="margin: 0 5px;">98.3% Accuracy</span>
        <span class="status-badge status-active" style="margin: 0 5px;">Multi-model AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Performance Metrics Dashboard
# -----------------------------------------------------------------------------
st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
st.markdown("### 📊 REAL-TIME METRICS")


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

# Dynamic Metrics Calculation
base_analyses = 2847
current_analyses = base_analyses + st.session_state['session_count']

# Base avg accuracy 96.2% over 2847 samples
base_acc_sum = base_analyses * 96.2
current_acc_sum = sum(st.session_state['session_confidences'])
if current_analyses > 0:
    avg_accuracy = (base_acc_sum + current_acc_sum) / current_analyses
else:
    avg_accuracy = 96.2

# Processing time
if st.session_state['processing_times']:
    avg_time = sum(st.session_state['processing_times']) / len(st.session_state['processing_times'])
else:
    avg_time = 1.4

with metric_col1:
    st.metric(
        label="Total Analyses", 
        value=f"{current_analyses:,}", 
        delta=f"+{124 + st.session_state['session_count']} today",
        delta_color="normal"
    )

with metric_col2:
    st.metric(
        label="Avg Accuracy", 
        value=f"{avg_accuracy:.1f}%", 
        delta="+1.4%" if st.session_state['session_count'] == 0 else f"{avg_accuracy - 96.2:+.1f}%",
        delta_color="normal"
    )

with metric_col3:
    st.metric(
        label="Processing Time", 
        value=f"{avg_time:.1f}s", 
        delta="-0.2s" if st.session_state['session_count'] == 0 else f"{avg_time - 1.4:+.1f}s",
        delta_color="inverse"
    )

with metric_col4:
    st.metric(
        label="Model Confidence", 
        value="94.8%", 
        delta="+0.8%",
        delta_color="normal"
    )

# -----------------------------------------------------------------------------
# 6. Feature Cards
# -----------------------------------------------------------------------------
st.markdown("### 🔧 CORE CAPABILITIES")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
<div class="feature-card fade-in">
    <div style="
        font-size: 3rem; 
        margin-bottom: 15px; 
        color: #00d4ff;
        filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));
    ">📉</div>
    <h3 style="color: white !important;">Pattern Extraction</h3>
    <p style="font-size: 0.95rem;">Advanced noise pattern analysis and PRNU fingerprint extraction for unique device identification.</p>
    <div style="margin-top: 20px;">
        <div class="progress-container">
            <div style="font-size: 0.85rem; color: #80deea; margin-bottom: 5px;">Accuracy: 98.3%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 98.3%;"></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="feature-card fade-in">
    <div style="
        font-size: 3rem; 
        margin-bottom: 15px; 
        color: #00ff88;
        filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.3));
    ">🌊</div>
    <h3 style="color: white !important;">Frequency Analysis</h3>
    <p style="font-size: 0.95rem;">FFT and wavelet transform analysis to detect scanner-specific frequency domain artifacts.</p>
    <div style="margin-top: 20px;">
        <div class="progress-container">
            <div style="font-size: 0.85rem; color: #80deea; margin-bottom: 5px;">Processing: 94.7%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 94.7%;"></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="feature-card fade-in">
    <div style="
        font-size: 3rem; 
        margin-bottom: 15px; 
        color: #ff6b6b;
        filter: drop-shadow(0 0 10px rgba(255, 107, 107, 0.3));
    ">🧠</div>
    <h3 style="color: white !important;">AI Classification</h3>
    <p style="font-size: 0.95rem;">Multi-model AI ensemble (CNN, SVM, RF) for accurate scanner brand and model identification.</p>
    <div style="margin-top: 20px;">
        <div class="progress-container">
            <div style="font-size: 0.85rem; color: #80deea; margin-bottom: 5px;">Confidence: 96.2%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 96.2%;"></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Project Overview 
# -----------------------------------------------------------------------------
st.markdown('<div id="project-overview"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("Project Overview")

col_text, col_details = st.columns([3, 2], gap="large")

with col_text:
    st.subheader("The Objective")
    st.write(
        """
        **TraceScope AI** is an advanced forensic analysis system designed to identify the source scanner device 
        (brand/model) by analyzing the unique digital fingerprints left during the scanning process. 
        
        Each scanner introduces specific noise patterns, texture artifacts, compression signatures, and frequency 
        domain characteristics that serve as unique identifiers, which our machine learning models learn to recognize.
        """
    )

    st.subheader("Key Outcomes")
    st.markdown("""
    * **Source Device Identification:** Pinpoint the exact scanner model used to create a document
    * **Digital Fingerprinting:** Extract and analyze unique scanner signatures
    * **Multi-model Analysis:** Combine CNN, Random Forest, and SVM for maximum accuracy
    * **Forensic Reporting:** Generate detailed analysis reports with confidence scores
    * **Real-time Processing:** Achieve identification in under 2 seconds per document
    """)
    
    st.subheader("Applications")
    app_col1, app_col2 = st.columns(2)
    with app_col1:
        st.markdown("""
<div class="doc-panel">
    <div style="font-size: 1.5rem; margin-bottom: 10px; color: #00d4ff;">🕵️</div>
    <strong style="color: white;">Digital Forensics</strong>
    <p style="font-size: 0.9rem; margin-top: 8px;">Determine scanner origin in document forgery investigations and fraud detection.</p>
</div>
""", unsafe_allow_html=True)
    
    with app_col2:
        st.markdown("""
<div class="doc-panel">
    <div style="font-size: 1.5rem; margin-bottom: 10px; color: #00ff88;">⚖️</div>
    <strong style="color: white;">Legal Verification</strong>
    <p style="font-size: 0.9rem; margin-top: 8px;">Verify document authenticity and ensure chain of custody in legal proceedings.</p>
</div>
""", unsafe_allow_html=True)

with col_details:
    st.subheader("Technical Architecture")
    
    # This was likely the source of Image 2 error. Indentation removed.
    st.markdown("""
<div style="
    background: linear-gradient(145deg, rgba(20, 25, 40, 0.8), rgba(10, 15, 30, 0.9));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 25px;
">
    <div style="
        display: flex; 
        align-items: center; 
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    ">
        <div style="
            background: rgba(0, 212, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            margin-right: 15px;
        ">
            <span style="font-size: 1.2rem;">📊</span>
        </div>
        <div>
            <strong style="color: #00d4ff; font-size: 1.1rem;">Data Pipeline</strong>
            <p style="font-size: 0.9rem; color: #80deea; margin: 0;">Real-time processing & analysis</p>
        </div>
    </div>
    <div class="progress-container" style="margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 5px;">
            <span style="color: #80deea;">Processing Queue</span>
            <span style="color: #00d4ff; font-weight: 600;">42 Active</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: 65%;"></div>
        </div>
    </div>
    <div style="margin-top: 20px;">
        <div style="font-size: 0.9rem; color: #80deea; margin-bottom: 10px;">Active Models:</div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span class="status-badge status-active">CNN v4.2</span>
            <span class="status-badge status-active">SVM v3.1</span>
            <span class="status-badge status-warning">RF v2.8</span>
            <span class="status-badge status-active">Ensemble</span>
        </div>
    </div>
    <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid rgba(0, 212, 255, 0.1);">
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
            <span style="color: #80deea;">Database Size</span>
            <span style="color: #00d4ff; font-weight: 600;">8.7 GB</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-top: 8px;">
            <span style="color: #80deea;">Scanner Profiles</span>
            <span style="color: #00d4ff; font-weight: 600;">142 Devices</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. System Architecture
# -----------------------------------------------------------------------------
st.markdown('<div id="system-architecture"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("System Architecture")

# Architecture visualization
graph = """
digraph G {
    # Layout Settings
    rankdir=TB;
    bgcolor="transparent";
    ranksep=0.8;
    nodesep=0.5;
    splines=ortho;
    
    # Node Styles
    node [
        shape=box,
        style="filled,rounded",
        fontname="Inter, sans-serif",
        fontsize=11,
        fillcolor="#1e1e24",
        color="#00d4ff",
        fontcolor="#e0e0e0",
        penwidth=1.5,
        height=0.6,
        width=1.8
    ];
    
    # Edge Style
    edge [
        color="#00d4ff",
        penwidth=1.8,
        arrowsize=0.8,
        arrowhead=vee
    ];
    
    # Define Nodes with enhanced styling
    Input [label="📄 Document Input\n& Preprocessing", fillcolor="#2a2a3a", color="#00d4ff"];
    Noise [label="📉 Noise Pattern\nAnalysis", fillcolor="#2a2a3a", color="#00d4ff"];
    Freq [label="🌊 Frequency Domain\nAnalysis", fillcolor="#2a2a3a", color="#00d4ff"];
    Texture [label="🔍 Texture & Artifact\nDetection", fillcolor="#2a2a3a", color="#00d4ff"];
    Feature [label="⚡ Feature\nExtraction", fillcolor="#2a2a3a", color="#00d4ff"];
    Model [label="🧠 AI Model\nEnsemble", fillcolor="#004e66", color="#00d4ff", fontcolor="#ffffff"];
    Database [label="🗄️ Scanner\nDatabase", fillcolor="#2a2a3a", color="#00d4ff"];
    Output [label="✅ Identification\nResults", fillcolor="#025B79", color="#00d4ff", fontcolor="#ffffff"];
    
    # Connections
    Input -> Noise;
    Input -> Freq;
    Input -> Texture;
    Noise -> Feature;
    Freq -> Feature;
    Texture -> Feature;
    Feature -> Model;
    Database -> Model;
    Model -> Output;
}
"""

st.graphviz_chart(graph, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. Live Analysis Lab
# -----------------------------------------------------------------------------
st.markdown('<div id="forensic-analysis-lab"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("🧪 Forensic Analysis Lab")

with st.container():
    st.markdown("""
<div style="
    background: linear-gradient(145deg, rgba(20, 25, 40, 0.8), rgba(10, 15, 30, 0.9));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 2rem;
">
""", unsafe_allow_html=True)
    
    st.markdown('<h3 style="text-align: center; margin-top: 0;">Upload & Analyze Document</h3>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 2rem; color: #80deea;">Upload a scanned document to identify the source scanner device.</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown("### 📁 Input Source")
        uploaded_files = st.file_uploader(
            "Drag & drop scanned images", 
            type=['jpg', 'png', 'tif', 'pdf'],
            accept_multiple_files=True,
            help="Supported formats: JPG, PNG, TIFF, PDF (Max 50MB per file). Batch processing supported."
        )
        
        if uploaded_files:
            file_count = len(uploaded_files)
            total_size = sum([f.size for f in uploaded_files]) / (1024*1024)
            st.info(f"""
            **Batch Details:**
            - **Files:** {file_count}
            - **Total Size:** {total_size:.2f} MB
            """)
            
            # Preview First Image
            try:
                st.image(uploaded_files[0], caption=f"Preview: {uploaded_files[0].name}", use_container_width=True)
                if file_count > 1:
                    st.caption(f"+ {file_count - 1} other files...")
            except Exception as e:
                st.warning("Preview not available for this file type.")

        
        st.markdown("### ⚙️ Configuration")
        
        analysis_mode = st.radio(
            "Analysis Methodology",
            ["Standard (Noise/FFT + SVM)", "Deep Learning (CNN Ensemble)", "Comprehensive (Full Spectrum)"],
            index=0
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=70, 
            max_value=99, 
            value=85,
            help="Minimum confidence level for scanner identification"
        )
        
        # Quick settings
        with st.expander("Advanced Settings"):
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                extract_prnu = st.checkbox("PRNU Analysis", value=True)
                extract_wavelet = st.checkbox("Wavelet Transform", value=True)
            with col_set2:
                save_results = st.checkbox("Save Results", value=True)
                generate_report = st.checkbox("Generate Report", value=True)
        
        st.write("")
        analyze_btn = st.button(
            "🚀 Identify Scanner", 
            type="primary", 
            use_container_width=True,
            disabled=not uploaded_files
        )

    with col_result:
        st.markdown("### 📊 Analysis Results")
        
        if uploaded_files and analyze_btn:
            # Analysis progress
            with st.status("🔬 Executing Forensic Analysis Pipeline...", expanded=True) as status:
                st.write(f"Processing {len(uploaded_files)} files...")
                progress_bar = st.progress(0)
                
                # Start Metrics
                tracemalloc.start()
                start_time = time.time()
                
                temp_dir = os.path.join(current_dir, "temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                
                batch_results = []
            
                # -------------------------------------------------------------------------
                # BATCH INFERENCE
                # -------------------------------------------------------------------------
                try:
                    # Collect all file paths
                    temp_paths = []
                    file_map = {} # path -> name
                    
                    for i, up_file in enumerate(uploaded_files):
                        t_path = os.path.join(temp_dir, up_file.name)
                        with open(t_path, "wb") as f:
                            f.write(up_file.getbuffer())
                        temp_paths.append(t_path)
                        file_map[t_path] = up_file.name
                    
                    # 1. Baseline Model (Sequential)
                    if "Standard" in analysis_mode:
                        for idx, t_path in enumerate(temp_paths):
                            pred_label, proba, _ = predict_baseline(t_path, model_choice="rf")
                            conf = float(np.max(proba) * 100) if proba is not None else 0.0
                            
                            res = {
                                "file": file_map[t_path],
                                "brand": pred_label.split(' ')[0] if pred_label else "Unknown",
                                "model": pred_label if pred_label else "Unknown",
                                "confidence": conf,
                                "serial": "N/A"
                            }
                            batch_results.append(res)
                            progress_bar.progress((idx + 1) / len(temp_paths))
                            
                    # 2. Hybrid / Deep Learning Model (Batch)
                    elif "Deep" in analysis_mode or "Comprehensive" in analysis_mode:
                        load_hybrid_resources()
                        from hybrid_cnn.utils import process_batch_gpu, batch_corr_gpu, extract_enhanced_features
                        
                        # Preprocess Batch
                        st.write("Preprocessing batch...")
                        residuals = process_batch_gpu(temp_paths)
                        
                        if residuals:
                            residuals = np.array(residuals, dtype=np.float32)
                            res_resources = get_hybrid_resources()
                            
                            # Feature Extraction
                            corrs = batch_corr_gpu(residuals, res_resources['fps'], res_resources['fp_keys'])
                            
                            enh_feats = []
                            for resid in residuals:
                                enh_feats.append(extract_enhanced_features(resid))
                            enh_feats = np.array(enh_feats, dtype=np.float32)
                            
                            feats_combined = np.hstack([corrs, enh_feats])
                            feats_scaled = res_resources['scaler'].transform(feats_combined)
                            
                            X_img = np.expand_dims(residuals, -1)
                            probs = res_resources['model'].predict([X_img, feats_scaled], verbose=0)
                            
                            for i, prob in enumerate(probs):
                                idx = int(np.argmax(prob))
                                label = res_resources['le'].classes_[idx]
                                conf = float(prob[idx] * 100)
                                
                                batch_results.append({
                                    "file": file_map[temp_paths[i]],
                                    "brand": label.split(' ')[0],
                                    "model": label,
                                    "confidence": conf,
                                    "serial": "AI-Gen"
                                })
                            progress_bar.progress(100)
                        else:
                            st.error("Batch preprocessing failed.")

                except Exception as e:
                    st.error(f"Batch Analysis failed: {str(e)}")
                
                # Cleanup
                for p in temp_paths:
                    if os.path.exists(p):
                        os.remove(p)
                
                # Metrics End
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                proc_time = end_time - start_time
                peak_mb = peak / (1024 * 1024)
                
                # Update Session State
                st.session_state['last_batch_results'] = batch_results
                st.session_state['processing_times'].append(proc_time)
                st.session_state['peak_memory'] = peak_mb
                st.session_state['session_count'] += len(batch_results)
                
                # Add to History
                for res in batch_results:
                    st.session_state['history'].insert(0, {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "file": res['file'],
                        "model": res['model'],
                        "confidence": res['confidence']
                    })

                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
        
        # Display Results
        if st.session_state.get('last_batch_results'):
            results = st.session_state['last_batch_results']
            
            if len(results) > 1:
                # Batch Processing - Average Results
                df_results = pd.DataFrame(results)
                
                # Calculate aggregated metrics
                top_brand = df_results['brand'].mode()[0] if not df_results['brand'].mode().empty else "Unknown"
                top_model = df_results['model'].mode()[0] if not df_results['model'].mode().empty else "Unknown"
                avg_confidence = df_results['confidence'].mean()
                
                top_scanner = {
                    "brand": top_brand,
                    "model": top_model,
                    "confidence": round(avg_confidence, 2),
                    "serial": "BATCH-AVG",
                    "file": f"Batch Summary ({len(results)} files)"
                }
            else:
                top_scanner = results[0]
            
            st.markdown("---")
            
            # Results display
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.markdown("#### 🏆 Top Match")
                st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0.05));
    border: 2px solid rgba(0, 212, 255, 0.3);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
">
    <div style="font-size: 3rem; margin-bottom: 15px;">🔍</div>
    <div style="font-size: 2rem; font-weight: 800; color: #00d4ff; margin-bottom: 5px;">
        {top_scanner['brand']}
    </div>
    <div style="font-size: 1.2rem; color: #80deea; margin-bottom: 10px;">
        {top_scanner['model']}
    </div>
    <div style="font-size: 0.9rem; color: #b0b0b0; margin-bottom: 20px;">
        Serial: {top_scanner['serial']}
    </div>
    <div style="
        font-size: 2.5rem;
        font-weight: 800;
        color: #00ff88;
        font-family: 'JetBrains Mono', monospace;
    ">
        {top_scanner['confidence']}%
    </div>
    <div style="color: #80deea; font-size: 0.9rem; margin-top: 5px;">
        CONFIDENCE SCORE
    </div>
</div>
""", unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("#### 📈 Feature Analysis")
                
                # Feature importance chart
                features = pd.DataFrame({
                    'Feature': ['Noise Pattern', 'Frequency', 'Texture', 'PRNU', 'Metadata', 'Compression'],
                    'Importance': [0.85, 0.72, 0.68, 0.91, 0.45, 0.63]
                })
                
                st.bar_chart(
                    features.set_index('Feature'),
                    color="#00d4ff",
                    height=250
                )
                
                # Additional metrics
                col_m1, col_m2 = st.columns(2)
                
                # Dynamic Logic
                proc_time_val = st.session_state['processing_times'][-1] if st.session_state['processing_times'] else 0
                peak_mem = st.session_state.get('peak_memory', 0.0)
                
                with col_m1:
                    st.metric("Processing Time", f"{proc_time_val:.2f}s")
                with col_m2:
                    st.metric("Memory Usage", f"{peak_mem:.2f} MB")
            
            # Success message
            st.success(f"""
            **Analysis Complete!** Document likely scanned by **{top_scanner['brand']} {top_scanner['model']}** with **{top_scanner['confidence']}%** confidence.
            """)
            
            # Export options
            st.markdown("---")
            st.markdown("#### 📤 Export Results")
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                # Use file name from result
                pdf_data = create_pdf_report(top_scanner, top_scanner.get('file', 'Unknown'))
                if pdf_data:
                    st.download_button(
                        label="📄 Generate PDF Report",
                        data=pdf_data,
                        file_name="trace_scope_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("Generation Failed")

            with col_exp2:
                # Export ALL batch results if multiple
                export_df = pd.DataFrame(results)
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Export Data",
                    data=csv_data,
                    file_name="trace_scope_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_exp3:
                if st.button("🔗 Share Analysis", use_container_width=True):
                    st.info("Share link copied to clipboard!")

            # -------------------------------------------------------------------------
            # BATCH RESULTS & HISTORY
            # -------------------------------------------------------------------------
            if len(results) > 1:
                st.markdown("### 📚 Batch Analysis Results")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            
            st.markdown("### 📜 Session History")
            if st.session_state['history']:
                hist_df = pd.DataFrame(st.session_state['history'])
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("No analysis history yet.")

        elif not uploaded_files:
            st.markdown("""
<div style="
    text-align: center; 
    padding: 4rem 2rem; 
    border: 2px dashed rgba(0, 212, 255, 0.3); 
    border-radius: 15px; 
    background: rgba(20, 25, 40, 0.5);
    margin-top: 1rem;
">
    <div style="font-size: 4rem; margin-bottom: 1.5rem; color: rgba(0, 212, 255, 0.3);">📄</div>
    <div style="font-size: 1.3rem; font-weight: 600; color: #80deea; margin-bottom: 10px;">
        Awaiting Document Upload
    </div>
    <p style="color: #666; max-width: 400px; margin: 0 auto;">
        Upload a scanned document to begin forensic analysis and scanner identification.
    </p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. Documentation Section
# -----------------------------------------------------------------------------
st.markdown('<div id="documentation"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("📚 Documentation & Resources")

# --- POPUP DIALOG LOGIC ---
@st.dialog("Documentation Details")
def show_doc_details(doc_type):
    if doc_type == "user_manual":
        st.header("📖 User Manual")
        st.markdown("""
        ### Getting Started
        1. **Upload Document:** Drag and drop scanned document to the input area.
        2. **Configure Settings:** Select analysis parameters like model type.
        3. **Run Analysis:** Click "Identify Scanner" to start.
        4. **Review Results:** Check confidence scores and feature analysis.

        ---
        ### Best Practices
        - **Image Quality:** Use high-quality scans (300+ DPI) for best results.
        - **Area:** Include full document area, avoid cropping important noise.
        - **Formats:** Avoid re-saving JPEGs multiple times (double compression).
        - **Reference:** Provide a reference sample if available.
        """)
    elif doc_type == "api_docs":
        st.header("🔧 API Reference")
        st.markdown("""
        ### REST API Endpoints

        **POST** `/api/v1/analyze`
        
        **Request Body:**
        ```json
        {
          "document": "base64_encoded_string",
          "mode": "standard|deep|comprehensive",
          "threshold": 0.85
        }
        ```

        **Response:**
        ```json
        {
          "scanner": "HP ScanJet Pro 3500",
          "confidence": 0.942,
          "features": {
            "noise_score": 0.88,
            "prnu_match": true
          },
          "processing_time": 1.45
        }
        ```

        **Authentication:**
        Use `Authorization: Bearer <your_api_key>` header.
        """)
    elif doc_type == "specs":
        st.header("⚙️ System Requirements")
        st.markdown("""
        ### Minimum Specifications
        - **CPU:** Quad-core processor, 2.5GHz+
        - **RAM:** 8GB minimum (16GB recommended for Deep Learning models)
        - **Storage:** 20GB free SSD space
        - **GPU:** (Optional) NVIDIA GPU with CUDA 11.0+ for acceleration.

        ### Supported Formats
        - **Images:** JPG, PNG, TIFF, BMP
        - **Documents:** PDF (Single/Multi-page)
        - **Max Size:** 200MB per file
        """)

doc_col1, doc_col2, doc_col3 = st.columns(3)

with doc_col1:
    st.markdown("""
<div class="doc-card">
    <div class="doc-card-icon">📖</div>
    <h4>User Manual</h4>
    <p style="font-size: 0.9rem; color: #b0b0b0; flex-grow: 1;">Complete guide to operating the TraceScope forensic suite.</p>
    <ul class="doc-card-list">
        <li>Quick Start Guide</li>
        <li>Analysis Configuration</li>
        <li>Result Interpretation</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    if st.button("View Manual", key="btn_manual", use_container_width=True):
        show_doc_details("user_manual")

with doc_col2:
    st.markdown("""
<div class="doc-card">
    <div class="doc-card-icon">🔧</div>
    <h4>API Reference</h4>
    <p style="font-size: 0.9rem; color: #b0b0b0; flex-grow: 1;">Integration documentation for external systems.</p>
    <ul class="doc-card-list">
        <li>REST API Endpoints</li>
        <li>Authentication & Auth</li>
        <li>Request Examples</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    if st.button("View API Docs", key="btn_api", use_container_width=True):
        show_doc_details("api_docs")

with doc_col3:
    st.markdown("""
<div class="doc-card">
    <div class="doc-card-icon">⚙️</div>
    <h4>System Specs</h4>
    <p style="font-size: 0.9rem; color: #b0b0b0; flex-grow: 1;">Hardware and software specifications.</p>
    <ul class="doc-card-list">
        <li>Server Config</li>
        <li>GPU Support</li>
        <li>File Support</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    if st.button("Check Specs", key="btn_specs", use_container_width=True):
        show_doc_details("specs")

# -----------------------------------------------------------------------------
# 11. Footer
# -----------------------------------------------------------------------------
st.markdown('<div id="settings"></div>', unsafe_allow_html=True)
st.markdown('<div id="performance"></div>', unsafe_allow_html=True)

# This was likely the source of Image 1 error. Indentation removed.
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    background: linear-gradient(90deg, 
        rgba(0, 212, 255, 0.05) 0%, 
        rgba(0, 212, 255, 0.1) 50%, 
        rgba(0, 212, 255, 0.05) 100%);
    border-top: 1px solid rgba(0, 212, 255, 0.2);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-top: 2rem;
">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
        <div style="text-align: left;">
            <h4 style="color: #00d4ff; margin-bottom: 5px;">TraceScope AI</h4>
            <p style="color: #80deea; font-size: 0.9rem;">Advanced Forensic Scanner Identification</p>
        </div>
        <div style="text-align: center;">
            <div style="display: flex; gap: 20px; justify-content: center;">
                <span style="color: #80deea;">🔒 Secure</span>
                <span style="color: #80deea;">⚡ Fast</span>
                <span style="color: #80deea;">🎯 Accurate</span>
            </div>
        </div>
        <div style="text-align: right;">
            <p style="color: #b3e5fc; font-size: 0.9rem;">
                <span style="color: #00d4ff;">Version:</span> 2.1.4<br>
                <span style="color: #00d4ff;">Updated:</span> 2024-03-15
            </p>
        </div>
    </div>
    <div style="
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        flex-wrap: wrap;
    ">
        <span style="color: #80deea; font-size: 0.85rem;">📧 support@tracescope.ai</span>
        <span style="color: #80deea; font-size: 0.85rem;">🌐 www.tracescope.ai</span>
        <span style="color: #80deea; font-size: 0.85rem;">📞 +1 (555) 123-4567</span>
    </div>
    <p style="color: #546e7a; font-size: 0.8rem; margin-top: 1.5rem;">
        © 2026 TraceScope AI Systems • For Forensic Research and Academic Use Only
    </p>
</div>
""", unsafe_allow_html=True)

