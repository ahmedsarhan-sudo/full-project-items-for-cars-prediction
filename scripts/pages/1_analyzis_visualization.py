import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 1. PAGE CONFIG
st.set_page_config(
    page_title='Pro Car Analytics',
    page_icon='🏎️',
    layout="wide"
)

# 2. ADVANCED CSS STYLING (Neural/3D Glassmorphism)
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* 3D Glass Cards for Figures and Tabs */
    .stPlotlyChart, .stImage, div[data-testid="stExpander"], .stTabs, .stDataFrame {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        padding: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        margin-bottom: 25px;
    }

    /* Neon Headlines */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0px 0px 15px rgba(0, 210, 255, 0.5);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 30, 0.95);
        border-right: 3px solid #00d2ff;
    }

    /* White text for better readability on dark cards */
    .stMarkdown p, .stMarkdown li {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# 3. ENHANCED 3D HERO SECTION
st.markdown("""
<div style="text-align: center; padding: 10px;">
    <h1 style="font-size: 3.8em; margin-bottom: 0;">🏎️ PRO-CAR ANALYTICS</h1>
    <p style="color: #00d2ff; letter-spacing: 5px; font-weight: bold;">DATA VISUALIZATION CORE</p>
    <div style="perspective: 1000px; padding: 20px;">
        <img src="https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1200&auto=format&fit=crop" 
             style="width: 85%; border-radius: 40px; 
                    transform: rotateX(15deg) rotateY(-5deg); 
                    box-shadow: 0 30px 60px rgba(0,210,255,0.4);
                    transition: transform 0.5s;">
    </div>
</div>
""", unsafe_allow_html=True)

# 4. GLOBAL MATPLOTLIB SETTINGS (For White Axis/Labels)
plt.rcParams.update({
    "figure.facecolor": "#161b22",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "text.color": "#ffffff",
    "grid.color": "#30363d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#ffffff"
})

# 5. DATA LOADING
if 'df' not in st.session_state:
    try:
        st.session_state['df'] = pd.read_csv(r'../datas/clean_car.csv')
    except:
        # Fallback empty df if path fails for demo
        st.session_state['df'] = pd.DataFrame()

if 'df_clean' not in st.session_state:
    try:
        st.session_state['df_clean'] = pd.read_csv(r'../datas/clean_car_filtering.csv')
    except:
        st.session_state['df_clean'] = pd.DataFrame()

# 6. APP LOGIC
if not st.session_state['df'].empty:
    df = st.session_state['df']
    df_filter = st.session_state['df_clean']

    type_of_visaul = st.sidebar.selectbox('Type of Visualization' , ['Select Field' , 'Static Visualization' , 'Dynamic Visualization'])
    
    if type_of_visaul == 'Select Field':
        st.info('Please select a visualization mode from the sidebar to begin analysis.')

    if type_of_visaul == 'Static Visualization' :
        tabs = st.tabs(['📊 Mfr vs Price' , '📈 Filtered View' , '🏢 Model Pricing' , '🔥 Top Models' , '⏳ Yearly Trends' , '🍕 Distribution'])

        with tabs[0]:
            st.markdown("### Manufacturer Value Analysis")
            col_a, col_b = st.columns(2)
            
            # Mean Price
            mean_price = df.groupby('Manufacturer')['Price'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(15,7))
            mean_price.plot(kind='bar', ax=ax, color='#00d2ff')
            ax.set_title("Manufacturers vs Mean Price", color='white', fontsize=18)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

            # Total Price
            Manufacturer_vs_price = df.groupby('Manufacturer')['Price'].sum().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(15,7))
            Manufacturer_vs_price.plot(kind='bar', ax=ax2, color='#ff0055')
            ax2.set_title("Manufacturers vs Total Price", color='white', fontsize=18)
            ax2.tick_params(axis='x', rotation=75)
            st.pyplot(fig2)
            
            # Countplot
            fig3, ax3 = plt.subplots(figsize=(15,7))
            sns.countplot(data=df, x='Manufacturer', order=df['Manufacturer'].value_counts().index, ax=ax3, palette="viridis")
            ax3.set_title("Manufacturers Count", color='white', fontsize=18)
            ax3.tick_params(axis='x', rotation=75)
            st.pyplot(fig3)

            comparison = pd.DataFrame({
                'Count': df['Manufacturer'].value_counts(),
                'Mean Price': df.groupby('Manufacturer')['Price'].mean(),
                'Total Revenue': df.groupby('Manufacturer')['Price'].sum()
            }).sort_values(by='Total Revenue', ascending=False)
            st.write("#### Detailed Comparison (Top 10)")
            st.dataframe(comparison.head(10), use_container_width=True)

        with tabs[1]:
            st.success('Displaying Filtered Core Manufacturers')
            fig, ax = plt.subplots(figsize=(15,6))
            sns.countplot(data=df_filter, y='Manufacturer', order=df_filter['Manufacturer'].value_counts().index, ax=ax, palette="magma")
            ax.set_title("Manufacturers Count (Filtered)", color='white')
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(13,6))
            Manufacturer_vs_price = df_filter.groupby('Manufacturer')['Price'].sum().sort_values(ascending=True)
            Manufacturer_vs_price.plot(kind='barh', ax=ax2, color='skyblue')
            ax2.set_title("Manufacturers vs Total Price (Filtered)", color='white')
            st.pyplot(fig2)

        with tabs[2]:
            st.markdown("### Top 9 Manufacturers Model Pricing")
            top_10_price = df_filter.groupby('Manufacturer')['Price'].sum().sort_values(ascending=False).head(10).index
            top_10_count = df_filter['Manufacturer'].value_counts().head(10).index
            union_top_9 = [i for i in top_10_price if i in top_10_count][:9]

            fig, axes = plt.subplots(3, 3, figsize=(14, 12))
            axes = axes.flatten()
            for ax, manufacturer in zip(axes, union_top_9):
                model_top_10 = df_filter[df_filter['Manufacturer'] == manufacturer]['Model'].value_counts().head(10).index
                df_temp = df_filter[df_filter['Model'].isin(model_top_10)].groupby('Model')['Price'].sum().sort_values(ascending=False)
                df_temp.plot(kind='bar', ax=ax, color='#00d2ff')
                ax.set_title(f"{manufacturer}'s model", color='white')
                ax.tick_params(axis='x', rotation=45)
            
            for i in range(len(union_top_9), len(axes)): fig.delaxes(axes[i])
            plt.tight_layout()
            st.pyplot(fig)

        with tabs[3]:
            st.markdown("### Model Volume Distribution")
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            for ax, manufacturer in zip(axes, union_top_9):
                model_top_10 = df[df['Manufacturer'] == manufacturer]['Model'].value_counts().head(10)
                sns.barplot(x=model_top_10.values, y=model_top_10.index, ax=ax, palette="coolwarm")
                ax.set_title(f"{manufacturer}'s Top Models", color='white', fontweight='bold')
            
            for i in range(len(union_top_9), len(axes)): fig.delaxes(axes[i])
            plt.tight_layout()
            st.pyplot(fig)

        with tabs[4]:
            st.markdown("### Production Year Revenue Trends (2010-2019)")
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            years = list(range(2010, 2020))
            for ax, manufacturer in zip(axes, union_top_9):
                df_yr = df[(df['Manufacturer'] == manufacturer) & (df['Prod. year'].between(2010, 2019))].groupby('Prod. year')['Price'].sum()
                ax.bar(range(len(df_yr)), df_yr.values, color='skyblue', alpha=0.6, label='Revenue')
                ax.plot(range(len(df_yr)), df_yr.values, color='royalblue', marker='o', linewidth=2, label='Trend')
                ax.set_title(f"{manufacturer}", color='white')
                ax.set_xticks(range(len(years)))
                ax.set_xticklabels(years, rotation=45)
            
            for i in range(len(union_top_9), len(axes)): fig.delaxes(axes[i])
            plt.tight_layout()
            st.pyplot(fig)

        with tabs[5]:
            st.markdown("### Categorical Breakdown")
            bie_char_list = ['Fuel type','Gear box type','Drive wheels','Wheel']
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for ax, col in zip(axes, bie_char_list):
                counts = df[col].value_counts().head(4)
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, explode=[0.05]*len(counts), shadow=True)
                ax.set_title(f"{col} Distribution", color='white', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

    if type_of_visaul == 'Dynamic Visualization':
        st.title("🚗 INTERACTIVE TREND DASHBOARD")

        Manufacturer_filt = st.sidebar.selectbox('Select Manufacturer', options=sorted(df_filter['Manufacturer'].unique()))
        option_model = df_filter[df_filter['Manufacturer'] == Manufacturer_filt]['Model'].unique()
        Model_filt = st.sidebar.selectbox('Select Model', options=option_model)
        
        # Action Center
        if st.sidebar.button('GENERATE ANALYTICS', key='show fig 1', use_container_width=True):
            cols = st.columns(2)

            with cols[0]:
                # Figure 1: Revenue Trend
                df_f1 = df_filter[(df_filter['Manufacturer'] == Manufacturer_filt) & (df_filter['Prod. year'].between(2010, 2019))]
                yearly = df_f1.groupby('Prod. year')['Price'].sum().reset_index()
                fig = px.bar(yearly, x='Prod. year', y='Price', title=f"{Manufacturer_filt} Revenue (2010-2019)", template="plotly_dark")
                fig.add_scatter(x=yearly['Prod. year'], y=yearly['Price'], mode='lines+markers', name='Trend')
                st.plotly_chart(fig, use_container_width=True)

            with cols[1]:
                # Figure 2: Model Revenue
                df_f2 = df_filter[df_filter['Manufacturer'] == Manufacturer_filt]
                yearly_m = df_f2.groupby('Model')['Price'].sum().sort_values(ascending=False).head(10).reset_index()
                fig2 = px.bar(yearly_m, x='Model', y='Price', title=f"{Manufacturer_filt} Models Revenue", template="plotly_dark")
                fig2.add_scatter(x=yearly_m['Model'], y=yearly_m['Price'], mode='lines+markers', name='Trend')
                st.plotly_chart(fig2, use_container_width=True)

            cols2 = st.columns(2)
            with cols2[0]:
                # Figure 3: Model Count
                counts = df_f1['Model'].value_counts().head(10).reset_index()
                fig3 = px.bar(counts, x='Model', y='count', title=f"{Manufacturer_filt} Models Count", template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)

            with cols2[1]:
                # Figure 4: Fuel Distribution
                fuel_data = df_f1['Fuel type'].value_counts().reset_index()
                fig4 = px.pie(fuel_data, names='Fuel type', values='count', hole=0.5, title="Fuel Type Split", template="plotly_dark")
                st.plotly_chart(fig4, use_container_width=True)

            cols3 = st.columns(2)
            with cols3[0]:
                # Figure 5: Specific Model Trend
                df_spec = df_f1[df_f1['Model'] == Model_filt].groupby('Prod. year')['Price'].sum().reset_index()
                fig5 = px.bar(df_spec, x='Prod. year', y='Price', title=f"{Model_filt} Trend", template="plotly_dark", color_discrete_sequence=['#00d2ff'])
                st.plotly_chart(fig5, use_container_width=True)

            with cols3[1]:
                # Figure 6: Model Fuel Split
                df_spec_fuel = df_f1[df_f1['Model'] == Model_filt]['Fuel type'].value_counts().reset_index()
                fig6 = px.pie(df_spec_fuel, names='Fuel type', values='count', hole=0.5, title=f"{Model_filt} Fuel Type", template="plotly_dark")
                st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("Please click 'GENERATE ANALYTICS' in the sidebar to load the dynamic charts.")

else:
    st.error("Data stream offline. Check file paths.")