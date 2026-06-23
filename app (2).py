import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E5D9E 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.85; }

    .prediction-box {
        background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box h2 { font-size: 1.3rem; margin: 0 0 0.5rem 0; opacity: 0.9; }
    .prediction-box h1 { font-size: 3rem; margin: 0; font-weight: 800; }
    .prediction-box p  { margin: 0.5rem 0 0 0; opacity: 0.85; }

    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #2E5D9E;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600 !important;
        color: #1F3864 !important;
    }
    .sidebar .sidebar-content { background: #f0f4f8; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model       = joblib.load("model.pkl")
        preprocessor= joblib.load("preprocessor.pkl")
        selector    = joblib.load("feature_selector.pkl")
        with open("metadata.json") as f:
            metadata = json.load(f)
        return model, preprocessor, selector, metadata, None
    except Exception as e:
        return None, None, None, None, str(e)

model, preprocessor, selector, metadata, load_error = load_artifacts()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💻 Laptop Price Predictor</h1>
    <p>Predict the price of any laptop based on its specifications using XGBoost ML Model</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"❌ Failed to load model artifacts: {load_error}")
    st.info("Make sure the models/ folder contains: model.pkl, preprocessor.pkl, feature_selector.pkl, metadata.json")
    st.stop()

# ── Sidebar — Input Form ──────────────────────────────────────
st.sidebar.markdown("## ⚙️ Laptop Specifications")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("### 🏢 Brand & Type")
    company = st.selectbox("Brand", [
        "Dell","Lenovo","HP","Asus","Acer","MSI","Toshiba",
        "Apple","Razer","Huawei","Xiaomi","Google","Samsung",
        "Microsoft","LG","Mediacom","Vero","Chuwi","Fujitsu","Panasonic"
    ])
    type_name = st.selectbox("Laptop Type", [
        "Notebook","Gaming","Ultrabook","2 in 1 Convertible",
        "Workstation","Netbook"
    ])

    st.markdown("---")
    st.markdown("### 🖥️ Display")
    inches = st.selectbox("Screen Size (inches)", [10.1,11.6,12.0,12.5,13.3,14.0,15.6,17.3], index=4)
    is_ips = st.checkbox("IPS Panel", value=True)
    is_touchscreen = st.checkbox("Touchscreen", value=False)
    res_width  = st.selectbox("Resolution Width",  [1366,1440,1600,1920,2560,3840], index=3)
    res_height = st.selectbox("Resolution Height", [768, 900, 1080,1200,1440,2160], index=2)

    st.markdown("---")
    st.markdown("### 🧮 Memory & Storage")
    ram_gb = st.select_slider("RAM (GB)", options=[2,4,6,8,12,16,24,32,64], value=8)
    storage_type = st.selectbox("Storage Type", ["SSD","HDD","Hybrid"])
    storage_gb   = st.select_slider("Storage (GB)", options=[32,64,128,256,512,1024,2048], value=256)

    st.markdown("---")
    st.markdown("### ⚙️ Processor & Graphics")
    cpu_brand = st.selectbox("CPU Brand", ["Intel","AMD","Apple","Samsung"])
    cpu_ghz   = st.slider("CPU Clock Speed (GHz)", 0.9, 4.5, 2.5, 0.1)
    gpu_brand = st.selectbox("GPU Brand", ["Intel","Nvidia","AMD","ARM"])

    st.markdown("---")
    st.markdown("### 💻 Other")
    os_clean = st.selectbox("Operating System", ["Windows","macOS","Linux","Chrome OS","No OS"])
    weight_kg = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

# ── Derived features ──────────────────────────────────────────
ppi = round(((res_width**2 + res_height**2)**0.5) / inches, 2)

ram_tier_map = {2:"Low (<=4GB)",4:"Low (<=4GB)",6:"Mid (8GB)",8:"Mid (8GB)",
                12:"High (16GB)",16:"High (16GB)",24:"Ultra (32GB+)",32:"Ultra (32GB+)",64:"Ultra (32GB+)"}
ram_tier = ram_tier_map.get(ram_gb, "Mid (8GB)")

storage_tier_map = {32:"Low (<=128GB)",64:"Low (<=128GB)",128:"Low (<=128GB)",
                    256:"Mid (256GB)",512:"High (512GB)",1024:"Ultra (1TB+)",2048:"Ultra (1TB+)"}
storage_tier = storage_tier_map.get(storage_gb, "Mid (256GB)")

premium_brands = ["Apple","Microsoft","Razer","LG"]
premium_types  = ["Gaming","Workstation","Ultrabook"]
is_premium = int(
    company in premium_brands or
    type_name in premium_types or
    ram_gb >= 16 or
    (storage_type == "SSD" and storage_gb >= 512)
)

weight_cat_map = lambda w: "Ultralight" if w<1.5 else ("Light" if w<2.0 else ("Standard" if w<2.5 else "Heavy"))
weight_category = weight_cat_map(weight_kg)

cpu_brand_score_map = {"Intel":1.0,"AMD":0.9,"Apple":1.2,"Samsung":0.7}
cpu_perf_score = round(cpu_brand_score_map.get(cpu_brand, 0.8) * cpu_ghz, 3)

# ── Build Input DataFrame ─────────────────────────────────────
input_data = pd.DataFrame([{
    "Company"       : company,
    "TypeName"      : type_name,
    "Inches"        : float(inches),
    "Ram_GB"        : ram_gb,
    "Weight_kg"     : weight_kg,
    "cpu_brand"     : cpu_brand,
    "cpu_ghz"       : cpu_ghz,
    "gpu_brand"     : gpu_brand,
    "storage_type"  : storage_type,
    "storage_gb"    : float(storage_gb),
    "res_width"     : float(res_width),
    "res_height"    : float(res_height),
    "ppi"           : ppi,
    "is_touchscreen": int(is_touchscreen),
    "is_ips"        : int(is_ips),
    "os_clean"      : os_clean,
    "ram_tier"      : ram_tier,
    "storage_tier"  : storage_tier,
    "is_premium"    : is_premium,
    "weight_category": weight_category,
    "cpu_perf_score": cpu_perf_score,
}])

# ── Main Layout ───────────────────────────────────────────────
col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown("### 📋 Selected Specifications")

    spec_col1, spec_col2, spec_col3 = st.columns(3)
    with spec_col1:
        st.markdown(f"""<div class="info-card">
            <b>Brand</b><br>{company}<br><br>
            <b>Type</b><br>{type_name}<br><br>
            <b>OS</b><br>{os_clean}
        </div>""", unsafe_allow_html=True)
    with spec_col2:
        st.markdown(f"""<div class="info-card">
            <b>Screen</b><br>{inches}" | {res_width}x{res_height}<br><br>
            <b>PPI</b><br>{ppi:.1f}<br><br>
            <b>Panel</b><br>{"IPS" if is_ips else "TN"} | {"Touch" if is_touchscreen else "No Touch"}
        </div>""", unsafe_allow_html=True)
    with spec_col3:
        st.markdown(f"""<div class="info-card">
            <b>RAM</b><br>{ram_gb}GB ({ram_tier})<br><br>
            <b>Storage</b><br>{storage_gb}GB {storage_type}<br><br>
            <b>Weight</b><br>{weight_kg}kg ({weight_category})
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="info-card" style="margin-top:1rem">
        <b>CPU:</b> {cpu_brand} @ {cpu_ghz}GHz &nbsp;|&nbsp;
        <b>GPU:</b> {gpu_brand} &nbsp;|&nbsp;
        <b>CPU Score:</b> {cpu_perf_score} &nbsp;|&nbsp;
        <b>Premium:</b> {"Yes" if is_premium else "No"}
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("### 💰 Price Prediction")
    predict_btn = st.button("🔍 Predict Price", use_container_width=True, type="primary")

    if predict_btn:
        try:
            # Preprocess
            X_proc = preprocessor.transform(input_data)
            X_sel  = selector.transform(X_proc)

            # Predict in log scale, inverse to EUR
            log_pred  = model.predict(X_sel)[0]
            price_eur = np.expm1(log_pred)

            # Price range (±15%)
            low  = price_eur * 0.85
            high = price_eur * 1.15

            st.markdown(f"""
            <div class="prediction-box">
                <h2>Estimated Price</h2>
                <h1>€{price_eur:,.0f}</h1>
                <p>Range: €{low:,.0f} — €{high:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence indicators
            st.markdown("#### 📊 Price Breakdown")
            if price_eur < 500:
                segment = "Budget"
                color   = "🟢"
            elif price_eur < 1000:
                segment = "Mid-Range"
                color   = "🟡"
            elif price_eur < 1800:
                segment = "Premium"
                color   = "🟠"
            else:
                segment = "Ultra Premium"
                color   = "🔴"

            st.metric("Market Segment", f"{color} {segment}")
            st.metric("Price (EUR)",    f"€{price_eur:,.0f}")
            st.metric("Price Range",    f"€{low:,.0f} – €{high:,.0f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.warning("Check that all model files are in the models/ folder.")

    else:
        st.info("👈 Set specifications in the sidebar, then click **Predict Price**")

# ── About Section ─────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About this App"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""**Model Details**
- Algorithm: XGBoost Regressor
- Features: 20 selected features
- Target: log(Price+1) → EUR
- Train/Test Split: 80/20""")
    with c2:
        st.markdown("""**Feature Engineering**
- PPI (display sharpness)
- CPU performance score
- RAM & storage tiers
- Premium laptop flag""")
    with c3:
        st.markdown("""**Dataset**
- Source: Kaggle Laptop Price
- ~1,300 laptops
- 13 raw features
- Brands: 20+""")
