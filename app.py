import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

from backend import detect_drift

# ================================================
# Page Config
# ================================================
st.set_page_config(
    page_title="DriftWatch",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================
# Theme State
# ================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Light Mode"

DARK = st.session_state.theme == "Dark Mode"

# ================================================
# Theme Variables
# ================================================
if DARK:
    bg_color          = "#0F1419"
    text_color        = "#F0F2F5"
    card_bg           = "#1E2533"
    accent            = "#8B5CF6"
    secondary_accent  = "#60A5FA"
    border_color      = "#303B52"
    metric_text       = "#60A5FA"
    chart_bg          = "#1E2533"
    hist_color1       = "#8B5CF6"
    hist_color2       = "#EC4899"
    plot_text         = "#F0F2F5"
else:
    bg_color          = "#F8FAFC"
    text_color        = "#0F1419"
    card_bg           = "#FFFFFF"
    accent            = "#6366F1"
    secondary_accent  = "#3B82F6"
    border_color      = "#E2E8F0"
    metric_text       = "#3B82F6"
    chart_bg          = "#FFFFFF"
    hist_color1       = "#6366F1"
    hist_color2       = "#EC4899"
    plot_text         = "#0F1419"


# ================================================
# CSS — separated from logic for maintainability
# ================================================
def inject_css():
    st.markdown(f"""
    <style>
    * {{ transition: background-color 0.3s ease, color 0.3s ease; }}

    .stApp {{
        background: {bg_color};
        color: {text_color};
    }}

    [data-testid="stAppViewContainer"] {{
        background: {bg_color};
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(16px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 0.6; }}
        50%       {{ opacity: 1; }}
    }}

    body, p, div {{ color: {text_color} !important; }}

    .main-title {{
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, {accent}, {secondary_accent});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        animation: fadeIn 0.8s ease-in-out;
        letter-spacing: -1px;
    }}

    .sub-text {{
        font-size: 16px;
        color: {text_color};
        opacity: 0.85;
        margin: 8px 0 0 0;
        animation: fadeIn 1.2s ease-in-out;
        font-weight: 500;
    }}

    .card {{
        padding: 24px;
        border-radius: 16px;
        background: {card_bg};
        border: 1px solid {border_color};
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-in-out;
        color: {text_color};
    }}

    .card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(124,58,237,0.15);
        border-color: {accent};
    }}

    .card-metric {{
        padding: 20px;
        border-radius: 14px;
        background: {card_bg};
        text-align: center;
        border: 1px solid {border_color};
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-in-out;
        color: {text_color};
    }}

    .card-metric:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.15);
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 700;
        margin: 24px 0 16px 0;
        color: {accent};
        animation: slideInRight 0.6s ease-in-out;
        padding-bottom: 8px;
        border-bottom: 2px solid {accent};
    }}

    .divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {border_color}, transparent);
        margin: 28px 0;
    }}

    .highlight {{
        padding: 14px 18px;
        border-radius: 10px;
        background: {accent}15;
        border-left: 4px solid {accent};
        color: {text_color};
        font-weight: 500;
        animation: fadeIn 1s ease-in-out;
    }}

    .highlight-success {{
        background: rgba(16,185,129,0.1);
        border-left-color: #10B981;
    }}

    .highlight-warning {{
        background: rgba(245,158,11,0.1);
        border-left-color: #F59E0B;
    }}

    .highlight-error {{
        background: rgba(239,68,68,0.1);
        border-left-color: #EF4444;
    }}

    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        border: 1px solid {accent} !important;
        background: linear-gradient(135deg, {accent}, {secondary_accent}) !important;
        color: white !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(99,102,241,0.3) !important;
    }}

    .stTabs [data-baseweb="tab-list"] button {{
        color: {text_color} !important;
        font-weight: 600 !important;
    }}

    .stTabs [aria-selected="true"] {{ color: {accent} !important; }}

    [data-testid="stMetricLabel"]  {{ color: {text_color} !important; font-weight: 600 !important; }}
    [data-testid="stMetricValue"]  {{ color: {metric_text} !important; }}

    .footer-wrapper {{
        margin-top: 50px;
        padding-top: 28px;
        border-top: 2px solid {border_color};
        text-align: center;
        animation: fadeIn 1.2s ease-in-out;
    }}

    .footer-heart {{ color: #EF4444; animation: pulse 1.5s ease-in-out infinite; }}

    @media (max-width: 768px) {{
        .main-title {{ font-size: 30px; }}
        .section-title {{ font-size: 18px; }}
    }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ================================================
# Helper: styled metric card
# ================================================
def metric_card(label: str, value: str, color: str = None):
    c = color or metric_text
    st.markdown(
        f'<div class="card-metric">'
        f'<b style="color:{text_color}">{label}</b><br>'
        f'<h2 style="margin:8px 0; color:{c}">{value}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ================================================
# Helper: matplotlib figure → bytes for download
# ================================================
def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ================================================
# Helper: build comparison charts
# ================================================
def make_histogram(train_col, new_col, feature: str):
    plt.style.use("dark_background" if DARK else "default")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=chart_bg)
    ax.set_facecolor(chart_bg)
    ax.hist(train_col, alpha=0.6, label="Training", bins=30, color=hist_color1, edgecolor="white", lw=0.5)
    ax.hist(new_col,   alpha=0.6, label="New Data", bins=30, color=hist_color2, edgecolor="white", lw=0.5)
    ax.legend(fontsize=10, facecolor=chart_bg, edgecolor=plot_text)
    ax.set_title(f"{feature} — Distribution", fontsize=12, fontweight="bold", color=plot_text)
    ax.set_xlabel("Value", fontsize=10, color=plot_text)
    ax.set_ylabel("Frequency", fontsize=10, color=plot_text)
    ax.grid(alpha=0.2, color=plot_text, linestyle="--")
    ax.tick_params(colors=plot_text)
    plt.tight_layout()
    return fig


def make_boxplot(train_col, new_col, feature: str):
    plt.style.use("dark_background" if DARK else "default")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=chart_bg)
    ax.set_facecolor(chart_bg)
    bp = ax.boxplot([train_col, new_col], labels=["Training", "New Data"],
                    patch_artist=True, widths=0.6)
    bp["boxes"][0].set(facecolor=hist_color1, alpha=0.7, edgecolor=plot_text, linewidth=1.5)
    bp["boxes"][1].set(facecolor=hist_color2, alpha=0.7, edgecolor=plot_text, linewidth=1.5)
    for el in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[el], color=plot_text, linewidth=1.5)
    ax.set_title(f"{feature} — Box Plot", fontsize=12, fontweight="bold", color=plot_text)
    ax.set_ylabel("Value", fontsize=10, color=plot_text)
    ax.grid(alpha=0.2, axis="y", color=plot_text, linestyle="--")
    ax.tick_params(colors=plot_text)
    plt.tight_layout()
    return fig


# ================================================
# HEADER
# ================================================
col_title, _, col_toggle = st.columns([3, 1, 0.4])

with col_title:
    st.markdown('<div class="main-title">📊 DriftWatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Smart Data Drift Detection & Monitoring</div>', unsafe_allow_html=True)

with col_toggle:
    st.markdown('<div style="padding-top:20px; text-align:right">', unsafe_allow_html=True)
    icon = "🌙" if not DARK else "☀️"
    if st.button(icon, key="theme_toggle", help="Toggle Dark/Light Mode"):
        st.session_state.theme = "Dark Mode" if not DARK else "Light Mode"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="highlight" style="background:{secondary_accent}10; border-left-color:{secondary_accent};">'
    f'🚀 Upload your dataset to detect and monitor data drift with interactive insights and visual analysis.</div>',
    unsafe_allow_html=True,
)

# ================================================
# File Upload
# ================================================
st.markdown('<div class="section-title">📂 Upload Your Dataset</div>', unsafe_allow_html=True)
st.markdown(f'<div class="highlight" style="background:{accent}10; border-left-color:{accent};">📌 Accepted format: CSV files only</div>', unsafe_allow_html=True)

file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

# ================================================
# Main Analysis
# ================================================
if file:
    data = pd.read_csv(file)

    # ---- File summary ----
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("📊 Rows",    str(len(data)))
    with c2: metric_card("📋 Columns", str(len(data.columns)))
    with c3: metric_card("✓ Status",   "Ready", color="#10B981")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Feature Drift", "📈 Visual Analysis"])

    # ================================================
    # TAB 1 — Overview
    # ================================================
    with tab1:
        st.markdown('<div class="section-title">👀 Dataset Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="highlight">🔍 <b>Ready to analyze?</b> Click below to run drift detection.</div>', unsafe_allow_html=True)

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            run = st.button("🚀 Run Drift Analysis", key="analyze_btn", use_container_width=True)
        with btn_col2:
            if "result" in st.session_state:
                if st.button("🗑 Clear Results", key="clear_btn", use_container_width=True):
                    del st.session_state["result"]
                    st.rerun()

        if run:
            with st.spinner("⏳ Analyzing data for drift patterns..."):
                try:
                    result = detect_drift(data)
                    st.session_state["result"] = result
                except ValueError as e:
                    st.error(f"❌ {e}")

    # ================================================
    # RESULTS
    # ================================================
    if "result" in st.session_state:
        result = st.session_state["result"]

        status_color = (
            "#EF4444" if result["status"] == "High Drift"
            else "#F59E0B" if result["status"] == "Medium Drift"
            else "#10B981"
        )

        # ---- Overview results ----
        with tab1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Drift Detection Results</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1: metric_card("📊 Drift Score",      f'{result["overall_score"]:.2f}')
            with m2: metric_card("🎯 Top Feature",      result["top_feature"],      color=metric_text)
            with m3: metric_card("⚡ Status",            result["status"],           color=status_color)
            with m4: metric_card("🔍 Features Analyzed", str(len(result["feature_drift"])))

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            if result["status"] == "High Drift":
                st.markdown('<div class="highlight highlight-error">🚨 <b>Alert:</b> High Drift Detected — Investigate data changes immediately.</div>', unsafe_allow_html=True)
            elif result["status"] == "Medium Drift":
                st.markdown('<div class="highlight highlight-warning">⚠️ <b>Warning:</b> Medium Drift Detected — Monitor data closely.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="highlight highlight-success">✅ <b>Good News:</b> Low Drift — Data remains stable.</div>', unsafe_allow_html=True)

        # ---- Feature Drift tab ----
        with tab2:
            st.markdown('<div class="section-title">📋 Individual Feature Drift Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="highlight">📊 Detailed drift metrics for each feature. <b>Severity</b> column summarises drift level per feature.</div>', unsafe_allow_html=True)

            drift_df = pd.DataFrame(result["feature_drift"]).T
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(drift_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Download drift report
            csv_bytes = drift_df.to_csv().encode("utf-8")
            st.download_button(
                label="⬇️ Download Drift Report (CSV)",
                data=csv_bytes,
                file_name="drift_report.csv",
                mime="text/csv",
            )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Statistical Comparison</div>', unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            with s1:
                st.markdown('<div class="card"><b>🎓 Training Dataset Statistics</b>', unsafe_allow_html=True)
                st.dataframe(result["train"].describe(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with s2:
                st.markdown('<div class="card"><b>🔍 New Dataset Statistics</b>', unsafe_allow_html=True)
                st.dataframe(result["new"].describe(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="highlight">💡 Differences in mean, std, min, max between training and new data indicate potential drift.</div>', unsafe_allow_html=True)

        # ---- Visual Analysis tab ----
        with tab3:
            st.markdown('<div class="section-title">📈 Visual Drift Analysis</div>', unsafe_allow_html=True)

            train = result["train"]
            new   = result["new"]

            # Let user pick any numeric feature, default to top drifting one
            numeric_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
            default_idx  = numeric_cols.index(result["top_feature"]) if result["top_feature"] in numeric_cols else 0
            feature = st.selectbox(
                "🔍 Select feature to visualise:",
                options=numeric_cols,
                index=default_idx,
            )

            st.markdown(f'<div class="highlight">Comparing <b>{feature}</b> between training and new data.</div>', unsafe_allow_html=True)

            if feature:
                v1, v2 = st.columns(2)

                with v1:
                    st.markdown('<div class="card"><b>📊 Distribution (Histogram)</b>', unsafe_allow_html=True)
                    fig1 = make_histogram(train[feature], new[feature], feature)
                    st.pyplot(fig1, use_container_width=True)
                    st.download_button("⬇️ Save Histogram", fig_to_bytes(fig1), f"{feature}_histogram.png", "image/png")
                    st.markdown('</div>', unsafe_allow_html=True)

                with v2:
                    st.markdown('<div class="card"><b>📦 Box Plot</b>', unsafe_allow_html=True)
                    fig2 = make_boxplot(train[feature], new[feature], feature)
                    st.pyplot(fig2, use_container_width=True)
                    st.download_button("⬇️ Save Box Plot", fig_to_bytes(fig2), f"{feature}_boxplot.png", "image/png")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="highlight highlight-success">💡 <b>How to read:</b> Significant changes in distribution shape, mean, or spread indicate data drift. The more the distributions differ, the higher the drift score.</div>', unsafe_allow_html=True)
            else:
                st.info("No numeric features available for visual analysis.")

else:
    st.markdown(
        f'<div class="highlight" style="text-align:center; padding:60px 20px; background:{card_bg}; border:2px dashed {border_color}; border-radius:16px;">'
        f'<h3 style="color:{text_color}; margin:0">👆 Start by uploading a CSV file</h3>'
        f'<p style="color:{text_color}; opacity:0.8; margin-top:10px">Drag and drop or click to select your dataset to begin drift detection</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ================================================
# Footer
# ================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="footer-wrapper">
    <p style="margin:0; color:{text_color}"><span class="footer-heart">❤️</span> Made with care for Data Quality</p>
    <p style="font-size:13px; opacity:0.7; margin-top:8px; color:{text_color}"><b>Thank you for using DriftWatch!</b> 🙏</p>
    <p style="font-size:11px; opacity:0.6; margin-top:10px; color:{text_color}">v1.1 | Data Drift Detection & Monitoring</p>
</div>
""", unsafe_allow_html=True)