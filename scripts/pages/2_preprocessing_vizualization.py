import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import pickle

# 1. PAGE CONFIG & CUSTOM THEME
st.set_page_config(
    page_title='Neural Car Analysis',
    page_icon='🏎️',
    layout="wide"
)

# 2. STRONG BACKGROUND & FRAMEWORK STYLING (Angular/Bootstrap Hybrid)
st.markdown("""
<style>
    /* Dark Deep Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* 3D Glass Cards for Figures */
    .stPlotlyChart, .stImage, div[data-testid="stExpander"], .stTabs {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        padding: 15px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }

    /* Angular Blue Headlines */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 10px rgba(0, 210, 255, 0.3);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 12, 41, 0.9);
        border-right: 2px solid #00d2ff;
    }

    /* Success/Error Alerts */
    .stAlert {
        border-radius: 50px;
        background: rgba(0, 210, 255, 0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 3. 3D HERO SECTION (Increased 3D Picture Style)
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="font-size: 3.5em;">🏁 PRO-CAR ANALYTICS CORE</h1>
    <img src="https://images.unsplash.com/photo-1614162692292-7ac56d7f7f1e?q=80&w=1200&auto=format&fit=crop" 
         style="width: 80%; border-radius: 30px; transform: perspective(1000px) rotateX(10deg); box-shadow: 0 20px 50px rgba(0,210,255,0.5); margin-bottom: 30px;">
    <p style="font-style: italic; color: #aaa;">Advanced Preprocessing & Model Performance Visualization</p>
</div>
""", unsafe_allow_html=True)

# 4. DATA & MODELS
@st.cache_resource
def load_data_and_models():
    # Attempt to load your specific paths
    try:
        with open(r'../models/xtrain.pkl' , 'rb') as f: xtrain = pickle.load(f)
        with open(r'../models/xtest.pkl' , 'rb') as f: xtest = pickle.load(f)
        with open(r'../models/ytrain.pkl' , 'rb') as f: ytrain = pickle.load(f)
        with open(r'../models/ytest.pkl' , 'rb') as f: ytest = pickle.load(f)
        with open(r'../models/ypred.pkl' , 'rb') as f: ypred = pickle.load(f)
        df = pd.read_csv(r'../datas/clean_car.csv')
        df_clean = pd.read_csv(r'../datas/clean_car_filtering.csv')
        return xtrain, xtest, ytrain, ytest, ypred, df, df_clean
    except:
        st.error("Data Path Error: Please ensure local pkl/csv files are in the correct directories.")
        return None

data_package = load_data_and_models()

if data_package:
    xtrain, xtest, ytrain, ytest, ypred, df, df_filter = data_package

    st.sidebar.image("https://images.unsplash.com/photo-1552519507-da3b142c6e3d?q=80&w=400&auto=format&fit=crop", caption="System Active")
    type_ = st.sidebar.selectbox('Select Preprocessing Stage', ['Overview', 'Model Visualization', 'Detection Outliers'])

    if type_ == 'Overview':
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success('✅ DATASTREAM DETECTED: Preprocessing Ready')
            st.write("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=600&auto=format&fit=crop", 
                     caption="Neural Analysis Engine", use_column_width=True)

    elif type_ == 'Model Visualization':
        st.markdown("## 📈 Performance Metrix")
        tabs = st.tabs(['🎨 Matplotlib Analytics', '⚡ Plotly Interactive'])

        with tabs[0]:
            y_test_array = np.array(ytest)
            sorted_index = np.argsort(y_test_array)
            y_test_sorted = y_test_array[sorted_index]
            
            fig, ax = plt.subplots(1, 2, figsize=(15,5), facecolor='#161b22')
            sns.scatterplot(x=np.arange(len(y_test_array)), y=y_test_array, alpha=0.6, ax=ax[0], color='#00d2ff')
            ax[0].set_facecolor('#0d1117')
            ax[0].set_title("Actual Prices", color='white')
            
            sns.scatterplot(x=np.arange(len(y_test_sorted)), y=y_test_sorted, alpha=0.4, ax=ax[1], color='#00d2ff')
            ax[1].set_facecolor('#0d1117')
            ax[1].set_title("Actual Prices (Sorted)", color='white')
            plt.tight_layout()
            st.pyplot(fig)

            # Residuals
            y_pred_array = np.array(ypred)
            residuals = y_test_array[sorted_index] - y_pred_array[sorted_index]
            fig2, ax2 = plt.subplots(1, 2, figsize=(15,5), facecolor='#161b22')
            sns.scatterplot(x=np.arange(len(residuals)), y=residuals, ax=ax2[0], color='#ff0055')
            ax2[0].axhline(y=0, color='green', linewidth=3)
            ax2[0].set_facecolor('#0d1117')
            ax2[0].set_title("Residuals (Error)", color='white')
            
            # Distance plotting logic
            ax2[1].scatter(y_test_array, y_pred_array, alpha=0.6, c='#00d2ff')
            ax2[1].plot([y_test_array.min(), y_test_array.max()], [y_test_array.min(), y_test_array.max()], 'r--')
            ax2[1].set_facecolor('#0d1117')
            ax2[1].set_title("Actual vs Predicted", color='white')
            st.pyplot(fig2)

        with tabs[1]:
            # Integrated Plotly (From your code)
            fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Car Prices", "Sorted Car Prices"))
            fig1.add_trace(go.Scatter(y=y_test_array, mode='markers', marker=dict(color='#00d2ff')), row=1, col=1)
            fig1.add_trace(go.Scatter(y=y_test_sorted, mode='markers', marker=dict(color='#00d2ff')), row=1, col=2)
            fig1.update_layout(template="plotly_dark", height=500, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

    elif type_ == 'Detection Outliers':
        st.markdown("## 🔍 Anomaly Detection")
        tabs = st.tabs(['Heatmap', 'Distribution (Log)', 'Pairplot', 'Deep Boxplots', 'Categorical vs Price'])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10,5), facecolor='#161b22')
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="mako", ax=ax)
            st.pyplot(fig)

        with tabs[1]:
            num_cols = df.select_dtypes(include='number').columns
            fig = plt.figure(figsize=(15, 12), facecolor='#161b22')
            outer = fig.add_gridspec(3, 3)
            for i, col in enumerate(num_cols[:9]):
                inner = outer[i].subgridspec(1, 2, wspace=0.3)
                ax1 = fig.add_subplot(inner[0], facecolor='#0d1117'); sns.kdeplot(df[col], ax=ax1, color='#00d2ff')
                ax2 = fig.add_subplot(inner[1], facecolor='#0d1117'); sns.kdeplot(np.log1p(df[col]), ax=ax2, color='#ff0055')
            st.pyplot(fig)

        with tabs[2]:
            st.write("### Multivariate Analysis (Sample 100)")
            pair_fig = sns.pairplot(df[:100], palette="mako")
            st.pyplot(pair_fig)

        with tabs[3]:
            # Your complex 4-column boxplot row
            num_col = ['Price', 'Levy', 'Engine volume', 'Mileage']
            fig, axes = plt.subplots(len(num_col), 4, figsize=(20, 15), facecolor='#161b22')
            for i, col in enumerate(num_col):
                sns.boxplot(x=df[col], ax=axes[i,0], color='#00d2ff')
                sns.boxplot(x=df[col], ax=axes[i,1], showfliers=False, color='#00d2ff')
                sns.boxenplot(x=df[col], ax=axes[i,2], color='#ff0055')
                sns.boxenplot(x=df[col], ax=axes[i,3], showfliers=False, color='#ff0055')
                for j in range(4): axes[i,j].set_facecolor('#0d1117')
            st.pyplot(fig)

        with tabs[4]:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='#161b22')
            axes = axes.flatten()
            for ax, col in zip(axes, ['Cylinders', 'Engine volume', 'Airbags']):
                top_categorical = df[col].value_counts().index[:10]
                sns.boxplot(x=df[df[col].isin(top_categorical)][col], y=df['Price'], ax=ax, palette="flare")
                ax.set_facecolor('#0d1117')
            st.pyplot(fig)