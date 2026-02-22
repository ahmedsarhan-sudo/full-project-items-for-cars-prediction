import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64

# Must be the very first Streamlit command
st.set_page_config(
    page_title="Car's Prediction | Neural Hub",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL LEGENDARY CSS ---
st.markdown("""
<style>
    /* Main Background override */
    .stApp {
        background-color: #050b14;
        color: #e2e8f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Hide Default UI */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Legendary Profile Card */
    .profile-container {
        background: rgba(16, 24, 39, 0.7);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 198, 255, 0.1);
        display: flex;
        align-items: center;
        border: 1px solid rgba(0, 198, 255, 0.2);
        transition: all 0.4s ease;
    }
    .profile-container:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 198, 255, 0.6);
        box-shadow: 0 15px 40px rgba(0, 198, 255, 0.2), inset 0 0 20px rgba(0, 198, 255, 0.2);
    }
    
    .profile-image {
        width: 170px;
        height: 170px;
        background-size: cover;
        background-position: center;
        border-radius: 50%;
        border: 4px solid #00c6ff;
        box-shadow: 0 0 25px rgba(0, 198, 255, 0.5);
        position: relative;
    }
    /* Pulsing ring around profile */
    .profile-image::after {
        content: ''; position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px;
        border-radius: 50%; border: 2px solid #00c6ff; opacity: 0.5;
        animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.9); opacity: 0.5; }
        100% { transform: scale(1.1); opacity: 0; }
    }

    .profile-text {
        margin-left: 40px;
        color: #f0f2f6;
    }
    .profile-text h2 {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #ffffff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
        letter-spacing: 1px;
    }
    .profile-text p {
        font-size: 1.1rem;
        margin: 5px 0;
        color: #94a3b8;
    }
    .badge {
        display: inline-block;
        padding: 5px 15px;
        background: rgba(0, 198, 255, 0.1);
        border: 1px solid #00c6ff;
        border-radius: 20px;
        color: #00c6ff;
        font-size: 0.9rem;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .social-links { margin-top: 20px; }
    .social-links a {
        color: #00c6ff;
        margin-right: 20px;
        transition: all 0.3s;
        text-decoration: none;
    }
    .social-links a:hover {
        color: #ffffff;
        text-shadow: 0 0 15px #00c6ff;
        transform: scale(1.1);
        display: inline-block;
    }

    /* System Cards */
    .card-container {
        display: flex; gap: 20px; margin-bottom: 40px;
    }
    .sys-card {
        background: rgba(16, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        flex: 1;
        text-align: center;
        transition: all 0.3s ease;
        overflow: hidden;
        position: relative;
    }
    .sys-card:hover {
        border-color: #00c6ff;
        transform: translateY(-5px);
        background: rgba(16, 24, 39, 0.9);
    }
    .sys-card img {
        width: 100%; height: 150px; object-fit: cover; border-radius: 10px; margin-bottom: 15px;
        opacity: 0.7; transition: opacity 0.3s;
    }
    .sys-card:hover img { opacity: 1; }
    .sys-card h3 { color: #fff; margin-bottom: 5px; font-size: 1.3rem;}
    .sys-card p { color: #94a3b8; font-size: 0.9rem;}

</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# --- IMAGE LOADING ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    image_base64 = get_base64_image(r"../profile_picture.jpg")
except FileNotFoundError:
    st.error("صورة الملف الشخصي (ahmed_sarhan.jpg) غير موجودة. يرجى التأكد من أنها في نفس مجلد التطبيق.")
    st.stop()


# --- 1. LEGENDARY PROFILE SECTION ---
with st.container():
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f'<div class="profile-image" style="background-image: url(data:image/jpeg;base64,{image_base64});"></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="profile-text">
                <h2>Ahmed Sarhan</h2>
                <div class="badge">LEAD DATA ANALYST</div>
                <p><i class="fas fa-university"></i> Faculty of Computer and Data Science</p>
                <p><i class="fas fa-microchip"></i> Architect of the Car Prediction Engine</p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/ahmed-sarhan-026b73359" target="_blank">
                        <i class="fab fa-linkedin fa-2x"></i>
                    </a>
                    <a href="https://github.com/ahmedsarhan-sudo" target="_blank">
                        <i class="fab fa-github fa-2x"></i>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


# --- 2. NEW PICTURES / MODULE SHOWCASE SECTION ---
st.markdown("<h3 style='color: #00c6ff; text-align: center; font-family: monospace; letter-spacing: 2px;'>// SYSTEM CAPABILITIES INITIALIZED</h3>", unsafe_allow_html=True)

st.markdown("""
<div class="card-container">
    <div class="sys-card">
        <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800&auto=format&fit=crop" alt="Data Analytics">
        <h3>Analysis Visualization</h3>
        <p>Deep dive into manufacturer trends, pricing matrices, and dynamic telemetry.</p>
    </div>
    <div class="sys-card">
        <img src="https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=800&auto=format&fit=crop" alt="Machine Learning">
        <h3>Preprocessing Engine</h3>
        <p>Advanced outlier detection, log transformations, and AI residual diagnostics.</p>
    </div>
    <div class="sys-card">
        <img src="https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=800&auto=format&fit=crop" alt="Supercar">
        <h3>Price Prediction</h3>
        <p>Neural network algorithms trained to predict automotive market values.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# --- 3. 3D HOLOGRAPHIC SCANNER & PARTICLES ---
# We keep your exact particles.js code but add a floating 3D car hologram in the center of it!
hologram_and_particles = """
<div id="particles-js"></div>
<div class="hologram-wrapper">
    <div class="hud-text">UPLOADING VEHICLE TELEMETRY...</div>
    <div class="car-container">
        <img src="https://images.unsplash.com/photo-1614162692292-7ac56d7f7f1e?q=80&w=1000&auto=format&fit=crop" alt="3D Car" class="floating-car">
        <div class="scan-line"></div>
    </div>
    <div class="hologram-base"></div>
</div>

<style>
    #particles-js { position: fixed; width: 100%; height: 100%; top: 0; left: 0; z-index: -1; }
    
    .hologram-wrapper {
        position: relative;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    
    .hud-text {
        color: #00c6ff; font-family: 'Courier New', monospace; font-size: 14px;
        letter-spacing: 3px; margin-bottom: 20px; animation: blink 2s infinite;
    }
    
    .car-container {
        position: relative; width: 700px; height: 350px;
        border-radius: 15px; overflow: hidden;
        box-shadow: 0 0 50px rgba(0, 198, 255, 0.15);
        border: 1px solid rgba(0, 198, 255, 0.3);
    }
    
    .floating-car {
        width: 100%; height: 100%; object-fit: cover;
        filter: brightness(0.9) contrast(1.2) hue-rotate(190deg) saturate(2);
        opacity: 0.8;
        mix-blend-mode: screen;
        animation: float 6s ease-in-out infinite;
    }
    
    .scan-line {
        position: absolute; left: 0; width: 100%; height: 4px;
        background: #00c6ff; box-shadow: 0 0 20px 5px #00c6ff;
        animation: scanning 3s linear infinite; opacity: 0.8;
    }
    
    .hologram-base {
        width: 500px; height: 40px; margin-top: -20px;
        background: radial-gradient(ellipse at center, rgba(0,198,255,0.4) 0%, transparent 70%);
        border-radius: 50%; filter: blur(5px);
    }
    
    @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    @keyframes scanning { 0% { top: -10%; } 100% { top: 110%; } }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>

<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 80, "density": {"enable": true, "value_area": 800}},
    "color": {"value": "#00c6ff"},
    "shape": {"type": "circle", "stroke": {"width": 0, "color": "#000"}},
    "opacity": {"value": 0.5},
    "size": {"value": 3},
    "line_linked": {"enable": true, "distance": 150, "color": "#00c6ff", "opacity": 0.4, "width": 1},
    "move": {"enable": true, "speed": 3, "direction": "none", "out_mode": "bounce"}
  },
  "interactivity": {
    "detect_on": "canvas",
    "events": {
      "onhover": {"enable": true, "mode": "grab"},
      "onclick": {"enable": true, "mode": "push"},
      "resize": true
    }
  },
  "retina_detect": true
});
</script>
"""
components.html(hologram_and_particles, height=600, scrolling=False)