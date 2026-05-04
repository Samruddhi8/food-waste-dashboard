import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Cabin Food Waste - ML Dashboard",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ══════════════════════════════════════════════════
   ARROW KILL — every possible selector for every
   Streamlit version
══════════════════════════════════════════════════ */
[data-testid="collapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNavCollapseButton"],
button[data-testid="stSidebarNavCollapseButton"],
.st-emotion-cache-h5rgaw,
.st-emotion-cache-1cypcdb,
.st-emotion-cache-dvne4q,
.eyeqlp53,
[aria-label="Close sidebar"],
[aria-label="Open sidebar"],
[aria-label="collapse navigation"],
[kind="header"] {
    display:         none !important;
    visibility:      hidden !important;
    opacity:         0 !important;
    pointer-events:  none !important;
    width:           0 !important;
    height:          0 !important;
    overflow:        hidden !important;
    position:        absolute !important;
    left:            -9999px !important;
}

/* ══════════════════════════════════════════════════
   GLOBAL FONT — Georgia for readability
══════════════════════════════════════════════════ */
html, body, [class*="css"], p, div, span, label,
h1, h2, h3, h4, li, td, th, button, input, select {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 15px;
}

/* ══════════════════════════════════════════════════
   BACKGROUND
══════════════════════════════════════════════════ */
.main {
    background: linear-gradient(135deg, #0a1628 0%, #1a2d4a 50%, #0f2540 100%);
    min-height: 100vh;
}
.block-container { padding: 2.5rem 2.5rem 2rem 2.5rem; }

/* ══════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f35 0%, #071525 100%) !important;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 14px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    padding: 5px 0;
    line-height: 1.8;
}
section[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }

/* ══════════════════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════════════════ */
.page-header {
    padding: 18px 0 10px 0;
    margin-bottom: 6px;
}
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.35;
    margin-bottom: 6px;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.page-subtitle {
    font-size: 1rem;
    color: #7eb8d8;
    margin-bottom: 0;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ══════════════════════════════════════════════════
   KPI CARDS
══════════════════════════════════════════════════ */
.kpi-card {
    background: linear-gradient(135deg, #1a3a6b 0%, #1e4d8c 100%);
    border: 1px solid #2d6abf;
    border-radius: 10px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(29,97,193,0.25);
    margin-bottom: 12px;
}
.kpi-card-green {
    background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(16,185,129,0.25);
    margin-bottom: 12px;
}
.kpi-card-gold {
    background: linear-gradient(135deg, #6b3a00 0%, #b45309 100%);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(245,158,11,0.25);
    margin-bottom: 12px;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 5px;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.kpi-label {
    font-size: 0.78rem;
    color: #a8c8e0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #6ee7b7;
    margin-top: 5px;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ══════════════════════════════════════════════════
   CONTENT CARDS
══════════════════════════════════════════════════ */
.content-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(100,160,220,0.2);
    border-radius: 10px;
    padding: 22px;
    margin-bottom: 16px;
}
.content-card h3 {
    color: #7dd3fc;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 14px;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.content-card p, .content-card li {
    color: #c8d8ea;
    font-size: 0.95rem;
    line-height: 1.75;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ══════════════════════════════════════════════════
   PIPELINE CARDS
══════════════════════════════════════════════════ */
.pipe-card {
    background: rgba(29,97,193,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    min-height: 110px;
}
.pipe-title {
    color: #93c5fd;
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 8px;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.pipe-desc {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.6;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ══════════════════════════════════════════════════
   ALERT BOXES
══════════════════════════════════════════════════ */
.alert-success {
    background: rgba(5,150,105,0.18);
    border-left: 4px solid #10b981;
    border-radius: 6px;
    padding: 13px 16px;
    margin: 10px 0;
    color: #a7f3d0;
    font-size: 0.93rem;
    line-height: 1.6;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.alert-info {
    background: rgba(37,99,168,0.18);
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 13px 16px;
    margin: 10px 0;
    color: #bfdbfe;
    font-size: 0.93rem;
    line-height: 1.6;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.alert-warning {
    background: rgba(180,83,9,0.18);
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 13px 16px;
    margin: 10px 0;
    color: #fde68a;
    font-size: 0.93rem;
    line-height: 1.6;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ══════════════════════════════════════════════════
   MISC
══════════════════════════════════════════════════ */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d6abf, transparent);
    margin: 18px 0;
}
.section-heading {
    color: #7dd3fc;
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 10px;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.section-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 14px;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    padding: 18px;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin-top: 2.5rem;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* ══════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    font-family: Georgia, 'Times New Roman', serif !important;
    padding: 8px 18px !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(29,97,193,0.45) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"]    { background: transparent !important; }

/* ══════════════════════════════════════════════════
   SLIDER & DATAFRAME
══════════════════════════════════════════════════ */
.stSlider label {
    color: #a8c8e0 !important;
    font-size: 0.95rem !important;
    font-family: Georgia, 'Times New Roman', serif !important;
}
.dataframe { font-size: 0.88rem !important; }
thead tr th {
    background: rgba(37,99,168,0.35) !important;
    color: #93c5fd !important;
    font-family: Georgia, 'Times New Roman', serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.04em !important;
}
tbody tr td { color: #c8d8ea !important; }
tbody tr:nth-child(even) td { background: rgba(255,255,255,0.025) !important; }
</style>
""", unsafe_allow_html=True)

# ── Chart theme
plt.rcParams.update({
    "figure.facecolor":  "#0d1f35",
    "axes.facecolor":    "#0d1f35",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1a2d4a",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "serif",
    "font.serif":        ["Georgia","Times New Roman","DejaVu Serif"],
})

C = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6",
     "#06b6d4","#f97316","#6366f1","#ec4899","#14b8a6"]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_MODELS = os.path.join(BASE_DIR, "models")
BASE_PROC   = os.path.join(BASE_DIR, "data", "processed")

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, n_actions))
    def forward(self, x): return self.net(x)

MULTIPLIERS = [0.70, 0.85, 1.00, 1.15]

def kpi(label, value, sub="", color="blue"):
    cls = {"green":"kpi-card-green","gold":"kpi-card-gold"}.get(color,"kpi-card")
    s   = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="{cls}"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>{s}</div>',
        unsafe_allow_html=True)

def success(t): st.markdown(f'<div class="alert-success">{t}</div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="alert-info">{t}</div>',    unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="alert-warning">{t}</div>', unsafe_allow_html=True)
def divider():  st.markdown('<div class="divider"></div>',            unsafe_allow_html=True)
def heading(t): st.markdown(f'<div class="section-heading">{t}</div>', unsafe_allow_html=True)
def subtext(t): st.markdown(f'<div class="section-sub">{t}</div>',     unsafe_allow_html=True)

def page_header(title, subtitle=""):
    sub = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title">{title}</div>'
        f'{sub}</div>',
        unsafe_allow_html=True)

@st.cache_data
def load_data():
    train  = pd.read_csv(os.path.join(BASE_PROC,"train_food.csv"),  parse_dates=["FlightDate"])
    test   = pd.read_csv(os.path.join(BASE_PROC,"test_food.csv"),   parse_dates=["FlightDate"])
    master = pd.read_csv(os.path.join(BASE_PROC,"master_food_features.csv"))
    with open(os.path.join(BASE_PROC,"pipeline_meta.json")) as f:
        meta = json.load(f)
    si = pd.read_csv(os.path.join(BASE_MODELS,"shap_importance.csv"))
    sv = np.load(os.path.join(BASE_MODELS,"shap_values_test.npy"))
    return train, test, master, meta, si, sv

@st.cache_resource
def load_models(feature_cols, train_df):
    hgb = joblib.load(os.path.join(BASE_MODELS,"model_hgb_tuned.pkl"))
    dqn_m = DQN(len(feature_cols), 4)
    dqn_m.load_state_dict(torch.load(
        os.path.join(BASE_MODELS,"dqn_policy.pth"), map_location="cpu"))
    dqn_m.eval()
    sc = StandardScaler()
    sc.fit(train_df[feature_cols])
    return hgb, dqn_m, sc

train, test, master, meta, shap_imp, shap_vals = load_data()
FEATURE_COLS = meta["FEATURE_COLS"]
hgb, dqn_model, scaler = load_models(FEATURE_COLS, train)

# ── Sidebar
with st.sidebar:
    st.markdown("## Cabin Food Waste")
    st.markdown("#### ML Dashboard")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "Home", "Data Overview", "Predictive Model",
        "SHAP Explainability", "Segmentation", "RL Optimisation"
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Records: 22,983")
    st.markdown("Flights: 1,200")
    st.markdown("Passengers: 300,103")
    st.markdown("Features: 69 pre-flight only")
    st.markdown("---")
    st.markdown("Zero data leakage. All features verified available before departure.")

# ══════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════
if page == "Home":
    page_header(
        "Predictive Modeling and Optimization<br>of Cabin Food Waste",
        "Master's Thesis | TU Braunschweig | IFF | Boeing"
    )
    divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Total Flights",    "1,200")
    with c2: kpi("Passengers",       "300,103")
    with c3: kpi("Waste Records",    "22,983")
    with c4: kpi("Mean Waste Rate",  "64.9%",    color="gold")
    with c5: kpi("Total Waste Cost", "EUR 4.3M", color="gold")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="content-card">
            <h3>Problem Statement</h3>
            <p>Airlines provision food before departure without knowing how much
            passengers will eat. All unsold food is incinerated after landing under
            international biosecurity regulations.</p>
            <br>
            <p>This creates a compounded problem:</p>
            <ul>
                <li>Food waste costs millions of euros per year in direct financial loss</li>
                <li>32 percent of negative passenger reviews mention food quality</li>
                <li>The issue is not just quantity - airlines are loading the wrong items</li>
                <li>Fixed loading rules cannot adapt to individual flight conditions</li>
            </ul>
            <br>
            <p>This pipeline uses machine learning built entirely from
            pre-flight data to predict waste and optimise loading decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="content-card"><h3>Key Results</h3>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Component": ["Predictive Model","Test R2","Test MAE",
                          "Train-Test Gap","Annual DQN Saving",
                          "Waste Reduction","DQN Stockout Rate"],
            "Result":    ["HistGBM Tuned","0.7467","0.1049",
                          "0.0002","EUR 1,029,104","13.8%","5.44%"]
        }), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    divider()
    heading("End-to-End Pipeline")

    pipeline = [
        ("NB01: Feature Engineering",
         "25 raw CSVs to 69 pre-flight features. Leakage policy verified mathematically."),
        ("NB02: Exploratory Analysis",
         "Waste distributions, route patterns, passenger profiles, feature correlations."),
        ("NB03: Predictive Model + SHAP",
         "6 models compared. HistGBM Tuned wins (R2 = 0.7467). SHAP on all 5,278 test records."),
        ("NB04: Segmentation",
         "K-Means: 4 passenger clusters, 3 flight clusters. Flight ARI = 1.0000."),
        ("NB05: RL Optimisation",
         "Q-Learning and DQN trained. DQN saves EUR 115,598 over 41-day test period."),
        ("NB07: Sentiment Analysis",
         "22,329 reviews analysed. 32 percent of negative reviews mention food terms.")
    ]
    cols = st.columns(3)
    for i,(title,desc) in enumerate(pipeline):
        with cols[i%3]:
            st.markdown(
                f'<div class="pipe-card"><div class="pipe-title">{title}</div>'
                f'<div class="pipe-desc">{desc}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DATA OVERVIEW
# ══════════════════════════════════════════════════════════
elif page == "Data Overview":
    page_header("Data Overview",
                "Exploratory analysis of 22,983 food waste records across 1,200 flights")
    divider()

    tab1,tab2,tab3 = st.tabs([
        "  Waste Distribution  ",
        "  Waste by Type and Round  ",
        "  Passenger Profile  "
    ])

    with tab1:
        left, right = st.columns([3,2])
        with left:
            fig,ax = plt.subplots(figsize=(7,4.5))
            n,_,_ = ax.hist(master["waste_rate"], bins=40,
                            color=C[0], edgecolor="#0d1f35", alpha=0.88)
            ax.axvline(master["waste_rate"].mean(), color=C[1], lw=2,
                       linestyle="--",
                       label=f"Mean: {master['waste_rate'].mean():.4f}")
            ax.axvline(master["waste_rate"].median(), color=C[2], lw=2,
                       linestyle="--",
                       label=f"Median: {master['waste_rate'].median():.4f}")
            ax.set_xlabel("Waste Rate"); ax.set_ylabel("Number of Records")
            ax.set_title("Waste Rate Distribution - 22,983 Records")
            ax.legend(facecolor="#0d1f35", edgecolor="#1e3a5f", fontsize=9)
            ax.fill_betweenx([0,n.max()], 0.5, 1.0, alpha=0.07, color=C[1])
            ax.text(0.75, n.max()*0.82, "76.6% above 0.5",
                    color=C[1], fontsize=9, ha="center")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            above50 = (master["waste_rate"]>0.5).mean()*100
            c1,c2 = st.columns(2)
            with c1:
                kpi("Mean",    f"{master['waste_rate'].mean():.4f}")
                kpi("Median",  f"{master['waste_rate'].median():.4f}")
                kpi("Min",     f"{master['waste_rate'].min():.4f}")
            with c2:
                kpi("Std Dev", f"{master['waste_rate'].std():.4f}")
                kpi("Max",     f"{master['waste_rate'].max():.4f}", color="gold")
                kpi("Above 0.5",f"{above50:.1f}%",                 color="gold")
            info("76.6% of records have waste rate above 0.5. High waste is the norm in this dataset.")

    with tab2:
        left, right = st.columns(2)
        with left:
            if "food_is_standard" in master.columns:
                def food_type(row):
                    if row.get("food_is_special",0):   return "Special Dietary"
                    if row.get("food_is_premium",0):   return "Premium"
                    if row.get("food_is_breakfast",0): return "Breakfast"
                    if row.get("food_is_children",0):  return "Children"
                    return "Standard"
                master["ftype"] = master.apply(food_type, axis=1)
                ts = (master.groupby("ftype")["waste_rate"].mean()
                      .reset_index().sort_values("waste_rate", ascending=True))
                fig,ax = plt.subplots(figsize=(6,4.5))
                ax.barh(ts["ftype"], ts["waste_rate"],
                        color=C[:len(ts)], edgecolor="#0d1f35", alpha=0.88)
                ax.set_xlabel("Mean Waste Rate")
                ax.set_title("Mean Waste Rate by Food Type")
                for i,v in enumerate(ts["waste_rate"]):
                    ax.text(v+0.003, i, f"{v:.3f}", va="center",
                            fontsize=9, color="#e2e8f0")
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            fig,ax = plt.subplots(figsize=(6,4.5))
            rounds = ["R1: Breakfast","R2: Main Meal","R4: Second Meal"]
            costs  = [0.312, 2.680, 1.233]
            bars   = ax.bar(rounds, costs, color=[C[0],C[1],C[2]],
                            edgecolor="#0d1f35", alpha=0.88, width=0.5)
            ax.set_ylabel("Total Waste Cost (EUR millions)")
            ax.set_title("Total Waste Cost by Service Round")
            for bar,v in zip(bars,costs):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.03,
                        f"EUR {v:.3f}M", ha="center", fontsize=9, color="#e2e8f0")
            plt.xticks(rotation=8); plt.tight_layout()
            st.pyplot(fig); plt.close()
        info("R2 (Main Meal) accounts for approximately 63 percent of total food waste cost.")

    with tab3:
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Total Passengers","300,103")
        with c2: kpi("Solo Travellers","97.5%")
        with c3: kpi("Economy Class","70.9%",color="green")
        with c4: kpi("Average Age","37 years")
        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            fig,ax = plt.subplots(figsize=(5,4.5))
            wedges,texts,auts = ax.pie(
                [70.9,14.0,13.3,1.8],
                labels=["Economy","Premium Economy","Business","First"],
                autopct="%1.1f%%", colors=C[:4], startangle=90,
                wedgeprops=dict(edgecolor="#0d1f35",linewidth=1.5))
            for at in auts: at.set_color("white"); at.set_fontsize(9)
            for t  in texts: t.set_color("#94a3b8"); t.set_fontsize(8.5)
            ax.set_title("Passengers by Cabin Class")
            st.pyplot(fig); plt.close()
        with right:
            info("97.5 percent of passengers travel solo. This near-homogeneous profile means passenger features contribute less individual predictive power than food item characteristics - confirmed by SHAP analysis.")

# ══════════════════════════════════════════════════════════
# PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════
elif page == "Predictive Model":
    page_header("Predictive Model Results",
                "Six models trained and evaluated on temporal test set - August to September 2025")
    divider()

    tab1,tab2 = st.tabs(["  Model Comparison  ","  Live Prediction  "])

    with tab1:
        md = pd.DataFrame({
            "Model": ["Dummy Mean","Linear Regression","Random Forest",
                      "Gradient Boosting","HistGradientBoosting",
                      "XGBoost","HistGBM Tuned"],
            "MAE":  [0.2578,0.1081,0.1058,0.1059,0.1056,0.1064,0.1049],
            "RMSE": [0.3151,0.1615,0.1594,0.1596,0.1587,0.1603,0.1586],
            "R2":   [-0.0001,0.7372,0.7442,0.7436,0.7465,0.7411,0.7467],
            "Train-Test Gap": [0.0,0.002,0.0075,0.006,0.0012,0.0474,0.0002]
        })
        dd = md.copy()
        for col in ["MAE","RMSE","R2","Train-Test Gap"]:
            dd[col] = dd[col].map(lambda x: f"{x:.4f}")
        dd["Status"] = [""]*6 + ["Selected"]
        st.dataframe(dd, use_container_width=True, hide_index=True)
        success("HistGBM Tuned: R2 = 0.7467, MAE = 0.1049, train-test gap = 0.0002. Near-zero gap confirms genuine generalisation with no overfitting.")

        left, right = st.columns(2)
        bc = [C[2] if m=="HistGBM Tuned" else C[0] for m in md["Model"]]
        with left:
            fig,ax = plt.subplots(figsize=(7,5.5))
            ax.barh(md["Model"], md["R2"], color=bc,
                    edgecolor="#0d1f35", alpha=0.88)
            ax.set_xlabel("R2 Score (higher is better)")
            ax.set_title("Model Comparison: R2 Score")
            ax.axvline(0, color="#64748b", lw=0.8)
            for i,v in enumerate(md["R2"]):
                ax.text(max(v,0)+0.004, i, f"{v:.4f}",
                        va="center", fontsize=9, color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            fig,ax = plt.subplots(figsize=(7,5.5))
            ax.barh(md["Model"], md["MAE"], color=bc,
                    edgecolor="#0d1f35", alpha=0.88)
            ax.set_xlabel("MAE (lower is better)")
            ax.set_title("Model Comparison: MAE")
            for i,v in enumerate(md["MAE"]):
                ax.text(v+0.002, i, f"{v:.4f}",
                        va="center", fontsize=9, color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        heading("Live Waste Rate Prediction")
        subtext("Select a record from the test set to see the trained model prediction.")

        idx  = st.slider("Test record index", 0, len(test)-1, 0)
        samp = test.iloc[idx]
        X_p  = samp[FEATURE_COLS].values.reshape(1,-1)
        pred = float(hgb.predict(X_p)[0])
        actual = float(samp["waste_rate"]) if "waste_rate" in samp.index else None

        left, right = st.columns(2)
        with left:
            rows = [{"Feature":f,
                     "Value": f"{samp[f]:.4f}" if isinstance(samp[f],float) else str(samp[f])}
                    for f in ["hist_wr_item","food_unit_cost","food_is_preorder",
                              "food_is_standard","food_is_premium","pax_avg_age","round_num"]
                    if f in samp.index]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with right:
            kpi("Predicted Waste Rate", f"{pred:.4f}")
            if actual is not None:
                err = abs(pred-actual)
                ca,cb = st.columns(2)
                with ca: kpi("Actual Waste Rate", f"{actual:.4f}")
                with cb: kpi("Absolute Error", f"{err:.4f}",
                             sub="Within MAE" if err<0.1049 else "Above MAE",
                             color="green" if err<0.1049 else "gold")
            uc  = float(samp.get("food_unit_cost",6.86))
            inv = float(samp["Inventory"]) if "Inventory" in samp.index else 10.0
            kpi("Predicted Waste Cost", f"EUR {pred*inv*uc:.2f}", color="gold")
            if pred > 0.75:   warn("High waste rate predicted. Loading reduction recommended.")
            elif pred < 0.40: success("Low waste rate predicted. Standard loading is appropriate.")
            else:             info("Moderate waste rate. Check SHAP page for feature-level drivers.")

# ══════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    page_header("SHAP Explainability Analysis",
                "SHAP values computed for all 5,278 test records using TreeExplainer. Base value: 0.6507.")
    divider()

    tab1,tab2 = st.tabs(["  Feature Importance  ","  Record Explanation  "])

    with tab1:
        left, right = st.columns([3,2])
        with left:
            top_n = st.slider("Top features to display", 10, min(30,len(shap_imp)), 20)
            top_df = shap_imp.head(top_n)
            fc_col, vc_col = top_df.columns[0], top_df.columns[1]
            fig,ax = plt.subplots(figsize=(8, top_n*0.42+1.2))
            col_b = [C[1] if i==0 else C[0] for i in range(len(top_df))]
            ax.barh(top_df[fc_col], top_df[vc_col], color=col_b,
                    edgecolor="#0d1f35", alpha=0.88)
            ax.invert_yaxis()
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_title(f"Top {top_n} Features by SHAP Importance")
            for i,v in enumerate(top_df[vc_col]):
                ax.text(v+0.001, i, f"{v:.4f}", va="center",
                        fontsize=8.5, color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            if len(shap_imp) > 0:
                kpi("Top Feature", str(shap_imp.iloc[0,0]))
                kpi("SHAP Value",  f"{float(shap_imp.iloc[0,1]):.4f}",
                    sub="More than 2x any other feature", color="gold")
            st.markdown("<br>", unsafe_allow_html=True)
            fig2,ax2 = plt.subplots(figsize=(5,4))
            gdata = {"Historical (2)":2,"Food Item (8)":8,
                     "Menu (4)":4,"Passenger (3)":3,"Flight (3)":3}
            wedges,texts,auts = ax2.pie(
                list(gdata.values()), labels=list(gdata.keys()),
                autopct="%1.0f%%", colors=C[:5], startangle=90,
                wedgeprops=dict(edgecolor="#0d1f35",linewidth=1.5))
            for at in auts: at.set_color("white"); at.set_fontsize(9)
            for t  in texts: t.set_color("#94a3b8"); t.set_fontsize(8)
            ax2.set_title("Feature Count per Group in Top 20")
            st.pyplot(fig2); plt.close()
            info("Only 2 historical features appear yet hist_wr_item alone (0.1725) exceeds all 8 food item features combined.")

        divider()
        st.dataframe(pd.DataFrame({
            "Feature":     ["hist_wr_item","food_is_standard",
                            "food_is_preorder","pax_avg_age"],
            "Effect":      ["High history: pushes prediction UP",
                            "Standard item: pushes prediction DOWN",
                            "Not preorder restricted: pushes UP",
                            "Older passengers: slightly DOWN"],
            "Linearity":   ["0.991","Binary","0.995","Weak"]
        }), use_container_width=True, hide_index=True)

    with tab2:
        heading("Individual Record Explanation")
        subtext("Select any test record to see which features drove its prediction.")

        rec_idx = st.slider("Test record", 0,
                            min(len(test)-1, shap_vals.shape[0]-1),
                            0, key="shap_rec")
        s      = test.iloc[rec_idx]
        X_r    = s[FEATURE_COLS].values.reshape(1,-1)
        pred_r = float(hgb.predict(X_r)[0])
        base   = 0.6507
        sv     = shap_vals[rec_idx]
        top_i  = np.argsort(np.abs(sv))[::-1][:12]
        top_f  = [FEATURE_COLS[i] for i in top_i]
        top_v  = [sv[i] for i in top_i]
        delta  = pred_r - base

        c1,c2,c3 = st.columns(3)
        with c1: kpi("Base Value",      f"{base:.4f}",  sub="Global mean prediction")
        with c2: kpi("Final Prediction", f"{pred_r:.4f}", color="gold")
        with c3: kpi("SHAP Shift",
                     f"{'+' if delta>0 else ''}{delta:.4f}",
                     sub="Above base" if delta>0 else "Below base",
                     color="gold" if delta>0 else "green")

        fig,ax = plt.subplots(figsize=(9,5.5))
        col_rec = [C[1] if v>0 else C[2] for v in top_v]
        ax.barh(top_f, top_v, color=col_rec, edgecolor="#0d1f35", alpha=0.88)
        ax.axvline(0, color="#64748b", lw=1.2)
        ax.set_xlabel("SHAP Value")
        ax.set_title(f"Record {rec_idx}: Top 12 Feature Contributions  |  "
                     f"Base {base:.4f} + shifts = {pred_r:.4f}")
        ax.invert_yaxis()
        plt.tight_layout(); st.pyplot(fig); plt.close()
        info("Red: pushed prediction above base (more waste). Green: pushed prediction below base (less waste).")

# ══════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════
elif page == "Segmentation":
    page_header("Passenger and Flight Segmentation",
                "K-Means clustering applied at two levels using pre-flight information only")
    divider()

    tab1,tab2 = st.tabs([
        "  Passenger Clusters (K=4)  ",
        "  Flight Clusters (K=3)  "
    ])

    with tab1:
        c1,c2,c3 = st.columns(3)
        with c1: kpi("Passengers Clustered","300,103")
        with c2: kpi("Silhouette Score","0.3118", sub="K = 4 selected", color="green")
        with c3: kpi("K-Means vs GMM ARI","0.7135")
        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(pd.DataFrame({
            "Cluster": ["0: Premium Travellers","1: Male Economy Solo",
                        "2: Female Economy Solo","3: Child and Family"],
            "Count":  ["44,631","125,327","126,370","3,775"],
            "Share":  ["14.9%","41.8%","42.1%","1.3%"],
            "Avg Age":["38.1","37.3","37.2","6.5"],
            "Key Characteristic": [
                "Business and First class mix, avg class ID 1.88",
                "100% male, Economy class, solo travel",
                "100% female, Economy class, solo travel",
                "Average age 6.5 years, family groups"
            ]
        }), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            fig,ax = plt.subplots(figsize=(5,4.5))
            wedges,texts,auts = ax.pie(
                [44631,125327,126370,3775],
                labels=["Premium\nTravellers","Male\nEconomy",
                        "Female\nEconomy","Child/Family"],
                autopct="%1.1f%%", colors=C[:4], startangle=90,
                wedgeprops=dict(edgecolor="#0d1f35",linewidth=1.5))
            for at in auts: at.set_color("white"); at.set_fontsize(9)
            for t  in texts: t.set_color("#94a3b8"); t.set_fontsize(8.5)
            ax.set_title("Passenger Cluster Sizes")
            st.pyplot(fig); plt.close()

        with right:
            fig,ax = plt.subplots(figsize=(5,4.5))
            ax.bar(["Premium","Male Eco","Female Eco","Child/Fam"],
                   [38.1,37.3,37.2,6.5], color=C[:4],
                   edgecolor="#0d1f35", alpha=0.88, width=0.55)
            ax.set_ylabel("Average Age")
            ax.set_title("Average Age by Cluster")
            for i,v in enumerate([38.1,37.3,37.2,6.5]):
                ax.text(i,v+0.3,str(v),ha="center",fontsize=10,color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        info("K-Means selected over GMM (silhouette 0.3118 vs 0.2870) because it preserves the Premium Travellers cluster. GMM merges this segment into economy clusters, losing the most operationally relevant group.")
        warn("Cluster proportions per flight showed low correlation with waste rate (-0.056 to 0.102). Passenger demographics do not strongly drive waste rate. Food item identity is the primary driver.")

    with tab2:
        c1,c2,c3 = st.columns(3)
        with c1: kpi("Flights Clustered","1,158")
        with c2: kpi("Silhouette Score","0.5666", sub="K = 3 selected", color="green")
        with c3: kpi("ARI vs Airline Labels","1.0000",
                     sub="Perfect reproduction", color="gold")
        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(pd.DataFrame({
            "Cluster": ["0: Economy Dominant","1: Premium Heavy","2: Mixed Economy"],
            "Flights": ["500 (43.2%)","339 (29.3%)","319 (27.5%)"],
            "Avg Passengers": ["226","275","261"],
            "Waste Rate":     ["0.654","0.653","0.654"],
            "Avg Waste Cost (EUR)": ["121","267","209"]
        }), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            fig,ax = plt.subplots(figsize=(5,4.5))
            ax.bar(["Economy\nDominant","Premium\nHeavy","Mixed\nEconomy"],
                   [121,267,209], color=C[:3],
                   edgecolor="#0d1f35", alpha=0.88, width=0.5)
            ax.set_ylabel("Avg Waste Cost per Record (EUR)")
            ax.set_title("Waste Cost by Flight Cluster")
            for i,v in enumerate([121,267,209]):
                ax.text(i,v+2,f"EUR {v}",ha="center",fontsize=10,color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            fig,ax = plt.subplots(figsize=(5,4.5))
            ax.bar(["Economy\nDominant","Premium\nHeavy","Mixed\nEconomy"],
                   [0.654,0.653,0.654], color=C[:3],
                   edgecolor="#0d1f35", alpha=0.88, width=0.5)
            ax.set_ylim(0,0.85)
            ax.set_ylabel("Average Waste Rate")
            ax.set_title("Waste Rate by Flight Cluster")
            for i,v in enumerate([0.654,0.653,0.654]):
                ax.text(i,v+0.01,str(v),ha="center",fontsize=10,color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        success("ARI = 1.0000: K-Means flight clusters perfectly reproduced the airline's own operational labels without ever seeing them. This validates the feature engineering pipeline.")
        info("Waste rate is identical across all clusters (0.654). Waste cost varies from EUR 121 to EUR 267 because premium flights carry more expensive food.")

# ══════════════════════════════════════════════════════════
# RL OPTIMISATION
# ══════════════════════════════════════════════════════════
elif page == "RL Optimisation":
    page_header("Reinforcement Learning Optimisation",
                "Four loading strategies evaluated on 5,278 test records - 41 days (August to September 2025)")
    divider()

    tab1,tab2,tab3 = st.tabs([
        "  Strategy Comparison  ",
        "  Stockout Analysis  ",
        "  Live DQN Recommendation  "
    ])

    with tab1:
        st.dataframe(pd.DataFrame({
            "Strategy":               ["Baseline (100%)","Always 70%","Q-Learning","DQN"],
            "Waste Cost (EUR)":       ["838,187","737,662","803,447","722,589"],
            "Saving vs Baseline (EUR)":["0","100,525","34,740","115,598"],
            "Saving (%)":             ["0.0%","12.0%","4.1%","13.8%"],
            "Annual Saving (EUR)":    ["0","894,915","309,271","1,029,104"],
            "Stockout Rate":          ["0.00%","20.33%","10.67%","5.44%"]
        }), use_container_width=True, hide_index=True)

        success("DQN achieves 13.8% waste cost reduction with a stockout rate of 5.44%. Annual saving: EUR 1,029,104.")

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("DQN Annual Saving",  "EUR 1,029,104", color="gold")
        with c2: kpi("DQN Waste Reduction","13.8%",         color="green")
        with c3: kpi("DQN Stockout Rate",  "5.44%",         color="green")
        with c4: kpi("Extra vs Always 70%","+EUR 134K/yr",  sub="DQN advantage")

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            fig,ax = plt.subplots(figsize=(6,4.5))
            costs = [838187,737662,803447,722589]
            bars  = ax.bar(
                ["Baseline\n100%","Always\n70%","Q-Learning","DQN"],
                [v/1000 for v in costs],
                color=[C[5],C[2],C[3],C[1]],
                edgecolor="#0d1f35", alpha=0.88, width=0.55)
            bars[3].set_edgecolor("#fbbf24"); bars[3].set_linewidth(2.5)
            ax.set_ylabel("Total Waste Cost (EUR thousands)")
            ax.set_title("Total Waste Cost by Strategy")
            for bar,v in zip(bars,costs):
                ax.text(bar.get_x()+bar.get_width()/2,
                        v/1000+3, f"EUR {v/1000:.0f}K",
                        ha="center", fontsize=9, color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            fig,ax = plt.subplots(figsize=(6,4.5))
            pcts  = [12.0,4.1,13.8]
            bars2 = ax.bar(
                ["Always 70%","Q-Learning","DQN"], pcts,
                color=[C[2],C[3],C[1]],
                edgecolor="#0d1f35", alpha=0.88, width=0.5)
            bars2[2].set_edgecolor("#fbbf24"); bars2[2].set_linewidth(2.5)
            ax.set_ylabel("Waste Cost Reduction (%)")
            ax.set_title("Percentage Saving vs Baseline")
            for bar,v in zip(bars2,pcts):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.2,
                        f"{v}%", ha="center", fontsize=11,
                        fontweight="bold", color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        divider()
        heading("DQN Action Distribution - What the Agent Learned")
        st.dataframe(pd.DataFrame({
            "Action":    ["Load 70%","Load 85%","Load 100%","Load 115%"],
            "DQN":       ["74.0%","5.2%","20.7%","0.0%"],
            "Always 70%":["100.0%","0.0%","0.0%","0.0%"],
            "Q-Learning":["54.0%","36.5%","0.6%","8.9%"]
        }), use_container_width=True, hide_index=True)
        info("The 20.7% of items kept at full loading by DQN is the key differentiator. These are low-waste items where cutting causes service failures that cost more than the waste saving.")

    with tab2:
        c1,c2,c3 = st.columns(3)
        with c1: kpi("Always 70% Stockout","20.33%", sub="1 in 5 decisions fail",  color="gold")
        with c2: kpi("Q-Learning Stockout", "10.67%", sub="1 in 9 decisions fail",  color="gold")
        with c3: kpi("DQN Stockout",        "5.44%",  sub="1 in 18 decisions fail", color="green")
        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            fig,ax = plt.subplots(figsize=(6,4.5))
            bars = ax.bar(
                ["Always 70%","Q-Learning","DQN"],
                [20.33,10.67,5.44],
                color=[C[1],C[3],C[2]],
                edgecolor="#0d1f35", alpha=0.88, width=0.5)
            ax.set_ylabel("Stockout Rate (%)")
            ax.set_title("Stockout Rate Comparison")
            for bar,v in zip(bars,[20.33,10.67,5.44]):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
                        f"{v}%", ha="center", fontsize=11,
                        fontweight="bold", color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with right:
            fig,ax = plt.subplots(figsize=(6,4.5))
            annual = [894915,309271,1029104]
            bars2  = ax.bar(
                ["Always 70%","Q-Learning","DQN"],
                [v/1000 for v in annual],
                color=[C[2],C[3],C[1]],
                edgecolor="#0d1f35", alpha=0.88, width=0.5)
            bars2[2].set_edgecolor("#fbbf24"); bars2[2].set_linewidth(2.5)
            ax.set_ylabel("Annual Saving (EUR thousands)")
            ax.set_title("Annualised Savings (scale factor 8.9)")
            for bar,v in zip(bars2,annual):
                ax.text(bar.get_x()+bar.get_width()/2,
                        v/1000+5, f"EUR {v/1000:.0f}K",
                        ha="center", fontsize=9, color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        info("A stockout occurs when the chosen loading falls below actual passenger consumption. The service penalty term (alpha = 0.5) guided DQN toward more conservative cuts on high-risk items.")

    with tab3:
        heading("Live DQN Loading Recommendation")
        subtext("Select any test record to see what the trained DQN agent recommends in real time.")

        rec = st.slider("Test record index", 0, len(test)-1, 42, key="rl_rec")
        s   = test.iloc[rec]
        X_r = s[FEATURE_COLS].values.reshape(1,-1)
        X_sc = scaler.transform(X_r)
        with torch.no_grad():
            qv = dqn_model(torch.FloatTensor(X_sc)).numpy()[0]
        action  = int(np.argmax(qv))
        mult    = MULTIPLIERS[action]
        pred_wr = float(hgb.predict(X_r)[0])
        uc      = float(s.get("food_unit_cost",6.86))
        inv     = float(s["Inventory"]) if "Inventory" in s.index else 10.0
        new_inv = inv*mult
        cost_n  = new_inv*pred_wr*uc
        cost_b  = inv*pred_wr*uc
        saving  = cost_b - cost_n

        left, right = st.columns(2)
        with left:
            rows = [{"Feature":f,
                     "Value": f"{s[f]:.4f}" if isinstance(s[f],float) else str(s[f])}
                    for f in ["hist_wr_item","food_unit_cost","food_is_preorder",
                              "food_is_standard","pax_avg_age"]
                    if f in s.index]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Action":   ["Load 70%","Load 85%","Load 100%","Load 115%"],
                "Q-Value":  [f"{v:.4f}" for v in qv],
                "Selected": ["YES" if i==action else "" for i in range(4)]
            }), use_container_width=True, hide_index=True)

        with right:
            kpi("Recommended Loading", f"{mult:.0%}", color="gold")
            ca,cb = st.columns(2)
            with ca: kpi("Standard Inventory",  f"{inv:.0f} units")
            with cb: kpi("Recommended Loading", f"{new_inv:.0f} units",
                         color="green" if mult<1.0 else "gold")
            kpi("Predicted Waste Rate",  f"{pred_wr:.4f}")
            kpi("Predicted Cost Saving", f"EUR {saving:.2f}",
                sub="vs baseline 100% loading",
                color="green" if saving>0 else "gold")

            if mult < 1.0:
                success(f"DQN recommends loading at {mult:.0%}. Predicted waste rate {pred_wr:.3f} - reducing inventory saves more than it risks in service penalties.")
            elif mult == 1.0:
                warn("DQN recommends full loading. This item has a low-waste profile where cutting would cause service failures costing more than the waste saving.")
            else:
                info(f"DQN recommends loading above standard at {mult:.0%}.")

# ── Footer
st.markdown("""
<div class="footer">
    Predictive Modeling and Optimization of Cabin Food Waste
    &nbsp;|&nbsp; Master's Thesis &nbsp;|&nbsp; TU Braunschweig
    &nbsp;|&nbsp; IFF &nbsp;|&nbsp; Boeing
    &nbsp;|&nbsp; All models trained on pre-flight data only
    &nbsp;|&nbsp; Zero data leakage
</div>
""", unsafe_allow_html=True)
