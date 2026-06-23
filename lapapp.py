import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1F3864 0%, #2E5D9E 100%);
    padding: 2rem; border-radius: 12px;
    margin-bottom: 2rem; text-align: center; color: white;
}
.main-header h1 { font-size: 2.2rem; margin: 0; }
.main-header p  { font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.85; }
.prediction-box {
    background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
    padding: 1.5rem; border-radius: 12px;
    text-align: center; color: white; margin: 0.5rem 0;
}
.prediction-box h2 { font-size: 1rem; margin: 0 0 0.5rem 0; opacity: 0.9; }
.prediction-box h1 { font-size: 2.2rem; margin: 0; font-weight: 800; }
.prediction-box p  { margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 0.85rem; }
.best-box {
    background: linear-gradient(135deg, #1F3864 0%, #2E5D9E 100%);
    padding: 2rem; border-radius: 12px;
    text-align: center; color: white; margin: 1rem 0;
}
.best-box h2 { font-size: 1.2rem; margin: 0 0 0.5rem 0; opacity: 0.9; }
.best-box h1 { font-size: 2.8rem; margin: 0; font-weight: 800; }
.best-box p  { margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
.info-card {
    background: #f8f9fa; border-left: 4px solid #2E5D9E;
    padding: 1rem 1.2rem; border-radius: 6px; margin: 0.5rem 0;
    color: #1F3864 !important;
}
.info-card b { color: #1F3864 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>💻 Laptop Price Predictor</h1>
    <p>Predict laptop price using 3 ML Models (Ridge, Random Forest, XGBoost) trained on 1300+ laptops</p>
</div>
""", unsafe_allow_html=True)

EUR_TO_INR = 90.0  # conversion rate

@st.cache_resource
def train_models():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from xgboost import XGBRegressor

    df = pd.read_csv('data/laptop_price_featured.csv')
    TARGET    = 'log_price'
    DROP_COLS = ['Price_euros','log_price','Ram','Weight','Cpu','Gpu','Memory','ScreenResolution','OpSys']
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing)
    y = df[TARGET]

    numerical_cols   = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    binary_cols      = [c for c in numerical_cols if X[c].nunique() == 2]
    numerical_cols   = [c for c in numerical_cols if c not in binary_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]), numerical_cols),
        ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                          ('enc',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), categorical_cols),
        ('bin','passthrough',binary_cols),
    ], remainder='drop')

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    selector = SelectKBest(f_regression, k=20)
    X_train_sel = selector.fit_transform(X_train_proc, y_train)
    X_test_sel  = selector.transform(X_test_proc)

    results = {}

    # 1. Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_sel, y_train)
    pred = ridge.predict(X_test_sel)
    results['Ridge Regression'] = {
        'model': ridge,
        'r2': r2_score(y_test, pred),
        'mae': mean_absolute_error(y_test, pred),
    }

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train_sel, y_train)
    pred = rf.predict(X_test_sel)
    results['Random Forest'] = {
        'model': rf,
        'r2': r2_score(y_test, pred),
        'mae': mean_absolute_error(y_test, pred),
    }

    # 3. XGBoost
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    xgb.fit(X_train_sel, y_train, eval_set=[(X_test_sel,y_test)], verbose=False)
    pred = xgb.predict(X_test_sel)
    results['XGBoost'] = {
        'model': xgb,
        'r2': r2_score(y_test, pred),
        'mae': mean_absolute_error(y_test, pred),
    }

    best_name = max(results, key=lambda k: results[k]['r2'])

    return results, best_name, preprocessor, selector, numerical_cols, categorical_cols, binary_cols

with st.spinner("Training 3 models... (first load ~45 sec)"):
    try:
        results, best_name, preprocessor, selector, numerical_cols, categorical_cols, binary_cols = train_models()
        st.success(f"All 3 models ready! Best model: **{best_name}** (R² = {results[best_name]['r2']:.3f})")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## Laptop Specifications")
company   = st.sidebar.selectbox("Brand", ["Dell","Lenovo","HP","Asus","Acer","MSI",
    "Toshiba","Apple","Razer","Huawei","Xiaomi","Google","Samsung","Microsoft","LG"])
type_name = st.sidebar.selectbox("Type", ["Notebook","Gaming","Ultrabook",
    "2 in 1 Convertible","Workstation","Netbook"])
inches    = st.sidebar.selectbox("Screen Size (inches)", [10.1,11.6,12.0,12.5,13.3,14.0,15.6,17.3], index=4)
is_ips         = st.sidebar.checkbox("IPS Panel", value=True)
is_touchscreen = st.sidebar.checkbox("Touchscreen", value=False)
res_width  = st.sidebar.selectbox("Resolution Width",  [1366,1440,1600,1920,2560,3840], index=3)
res_height = st.sidebar.selectbox("Resolution Height", [768,900,1080,1200,1440,2160], index=2)
ram_gb     = st.sidebar.select_slider("RAM (GB)", options=[2,4,6,8,12,16,24,32,64], value=8)
storage_type = st.sidebar.selectbox("Storage Type", ["SSD","HDD","Hybrid"])
storage_gb   = st.sidebar.select_slider("Storage (GB)", options=[32,64,128,256,512,1024,2048], value=256)
cpu_brand = st.sidebar.selectbox("CPU Brand", ["Intel","AMD","Apple","Samsung"])
cpu_ghz   = st.sidebar.slider("CPU Speed (GHz)", 0.9, 4.5, 2.5, 0.1)
gpu_brand = st.sidebar.selectbox("GPU Brand", ["Intel","Nvidia","AMD","ARM"])
os_clean  = st.sidebar.selectbox("OS", ["Windows","macOS","Linux","Chrome OS","No OS"])
weight_kg = st.sidebar.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

# ── Derived features ──────────────────────────────────────────
ppi = round(((res_width**2 + res_height**2)**0.5) / float(inches), 2)
ram_tier_map = {2:"Low (<=4GB)",4:"Low (<=4GB)",6:"Mid (8GB)",8:"Mid (8GB)",
                12:"High (16GB)",16:"High (16GB)",24:"Ultra (32GB+)",32:"Ultra (32GB+)",64:"Ultra (32GB+)"}
ram_tier = ram_tier_map.get(ram_gb,"Mid (8GB)")
storage_tier_map = {32:"Low (<=128GB)",64:"Low (<=128GB)",128:"Low (<=128GB)",
                    256:"Mid (256GB)",512:"High (512GB)",1024:"Ultra (1TB+)",2048:"Ultra (1TB+)"}
storage_tier = storage_tier_map.get(storage_gb,"Mid (256GB)")
premium_brands = ["Apple","Microsoft","Razer","LG"]
premium_types  = ["Gaming","Workstation","Ultrabook"]
is_premium = int(company in premium_brands or type_name in premium_types
                 or ram_gb >= 16 or (storage_type=="SSD" and storage_gb>=512))
weight_cat = ("Ultralight" if weight_kg<1.5 else "Light" if weight_kg<2.0
              else "Standard" if weight_kg<2.5 else "Heavy")
cpu_score_map = {"Intel":1.0,"AMD":0.9,"Apple":1.2,"Samsung":0.7}
cpu_perf_score = round(cpu_score_map.get(cpu_brand,0.8)*cpu_ghz, 3)

input_data = pd.DataFrame([{
    "Company":company,"TypeName":type_name,"Inches":float(inches),
    "Ram_GB":ram_gb,"Weight_kg":weight_kg,"cpu_brand":cpu_brand,
    "cpu_ghz":cpu_ghz,"gpu_brand":gpu_brand,"storage_type":storage_type,
    "storage_gb":float(storage_gb),"res_width":float(res_width),
    "res_height":float(res_height),"ppi":ppi,
    "is_touchscreen":int(is_touchscreen),"is_ips":int(is_ips),
    "os_clean":os_clean,"ram_tier":ram_tier,"storage_tier":storage_tier,
    "is_premium":is_premium,"weight_category":weight_cat,
    "cpu_perf_score":cpu_perf_score,
}])

# ── Main layout ───────────────────────────────────────────────
st.markdown("### Selected Specifications")
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="info-card"><b>Brand</b><br>{company}<br><br>'
                f'<b>Type</b><br>{type_name}<br><br><b>OS</b><br>{os_clean}</div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="info-card"><b>Screen</b><br>{inches}" {res_width}x{res_height}<br><br>'
                f'<b>PPI</b><br>{ppi:.1f}<br><br>'
                f'<b>Panel</b><br>{"IPS" if is_ips else "TN"} | {"Touch" if is_touchscreen else "No Touch"}</div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="info-card"><b>RAM</b><br>{ram_gb}GB<br><br>'
                f'<b>Storage</b><br>{storage_gb}GB {storage_type}<br><br>'
                f'<b>Weight</b><br>{weight_kg}kg</div>',
                unsafe_allow_html=True)
st.markdown(f'<div class="info-card" style="margin-top:1rem">'
            f'<b>CPU:</b> {cpu_brand} @ {cpu_ghz}GHz &nbsp;|&nbsp;'
            f'<b>GPU:</b> {gpu_brand} &nbsp;|&nbsp;'
            f'<b>Premium:</b> {"Yes" if is_premium else "No"}</div>',
            unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Price Prediction — All 3 Models")

if st.button("Predict Price", use_container_width=True, type="primary"):
    try:
        X_proc = preprocessor.transform(input_data)
        X_sel  = selector.transform(X_proc)

        preds_inr = {}
        for name, info in results.items():
            price_eur = np.expm1(info['model'].predict(X_sel)[0])
            preds_inr[name] = price_eur * EUR_TO_INR

        # Best model highlighted first
        best_price = preds_inr[best_name]
        low_inr, high_inr = best_price*0.85, best_price*1.15
        segment = ("Budget" if best_price<45000 else
                   "Mid-Range" if best_price<90000 else
                   "Premium" if best_price<160000 else "Ultra Premium")

        st.markdown(f"""
        <div class="best-box">
            <h2>Best Model Prediction — {best_name}</h2>
            <h1>₹{best_price:,.0f}</h1>
            <p>Range: ₹{low_inr:,.0f} — ₹{high_inr:,.0f} &nbsp;|&nbsp; Segment: {segment}</p>
            <p style="font-size:0.85rem; opacity:0.75">≈ €{best_price/EUR_TO_INR:,.0f} EUR &nbsp;|&nbsp; R² = {results[best_name]['r2']:.3f}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Comparison across all 3 models")
        cols = st.columns(3)
        for col, (name, info) in zip(cols, results.items()):
            price = preds_inr[name]
            with col:
                marker = " 🏆" if name == best_name else ""
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{name}{marker}</h2>
                    <h1>₹{price:,.0f}</h1>
                    <p>≈ €{price/EUR_TO_INR:,.0f} EUR</p>
                    <p>R² = {info['r2']:.3f} | MAE = {info['mae']:.3f}</p>
                </div>""", unsafe_allow_html=True)

        comp_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Predicted Price (INR)': [f"₹{preds_inr[n]:,.0f}" for n in results],
            'Predicted Price (EUR)': [f"€{preds_inr[n]/EUR_TO_INR:,.0f}" for n in results],
            'Test R²': [f"{results[n]['r2']:.3f}" for n in results],
            'Test MAE (log scale)': [f"{results[n]['mae']:.3f}" for n in results],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("Set specs in sidebar then click Predict Price")

st.markdown("---")
with st.expander("About this App"):
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Models Trained**\n- Ridge Regression\n- Random Forest\n- XGBoost\n- 20 selected features each\n- 80/20 train-test split")
    with c2:
        st.markdown("**Features**\n- PPI display score\n- CPU perf score\n- RAM/storage tiers\n- Premium flag")
    with c3:
        st.markdown("**Dataset**\n- Kaggle Laptop Price\n- ~1300 laptops\n- 20+ brands\n- EUR to INR @ 90")
