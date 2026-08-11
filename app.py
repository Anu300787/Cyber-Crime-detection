import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.express as px

from tensorflow.keras.models import load_model


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Threat Monitoring Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CYBERSECURITY DARK THEME
# =====================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(circle at top, #0f172a, #020617);
        color: white;
    }

    .block-container {
        padding-top: 1rem;
    }

    h1, h2, h3 {
        color: #22d3ee;
    }

    .cyber-card {
        background:
        linear-gradient(
            145deg,
            #111827,
            #020617
        );

        border-radius: 18px;

        padding: 18px;

        border:
        1px solid #164e63;

        box-shadow:
        0 0 18px rgba(34, 211, 238, 0.15);

        text-align: center;
    }

    .online {
        background: #052e16;

        border:
        1px solid #22c55e;

        border-radius: 15px;

        padding: 12px;

        text-align: center;

        font-size: 20px;

        font-weight: bold;
    }

    .danger {
        background: #450a0a;

        border:
        1px solid #ef4444;

        border-radius: 15px;

        padding: 12px;

        text-align: center;

        font-size: 22px;

        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    # 🛡️ AI THREAT MONITORING CENTER

    ### LSTM Autoencoder Network Intrusion Detection
    """
)


# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =====================================================
# MODEL / ARTIFACT PATHS
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm_autoencoder_final.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "threshold.pkl"
)


# =====================================================
# LOAD MODEL + ARTIFACTS
# =====================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        os.path.join(BASE_DIR, "models", "lstm_autoencoder_final.keras"),
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)

    threshold = joblib.load(THRESHOLD_PATH)

    return model, scaler, threshold


# =====================================================
# LOAD ARTIFACTS
# =====================================================

try:

    model, scaler, threshold = load_artifacts()

except Exception as e:

    st.error(
        "❌ Failed to load AI model or required artifacts."
    )

    st.exception(e)

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ SYSTEM INFO")

    st.info(
        """
        Algorithm:
        LSTM Autoencoder

        Dataset:
        CICIDS2017

        Sequence:
        30 Timesteps

        Features:
        68

        Detection:
        Reconstruction Error
        """
    )

    st.success(
        "SYSTEM ACTIVE"
    )


# =====================================================
# SEQUENCE CREATION
# =====================================================

def create_sequences(
    X,
    sequence_length
):

    sequences = []

    for i in range(
        len(X) - sequence_length + 1
    ):

        sequences.append(
            X[i:i + sequence_length]
        )

    return np.array(
        sequences,
        dtype=np.float32
    )


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📂 Upload CICIDS2017 CSV",
    type="csv"
)


if uploaded_file is None:

    st.warning(
        "Upload dataset to start AI analysis."
    )

    st.stop()


# =====================================================
# READ DATASET
# =====================================================

df = pd.read_csv(
    uploaded_file
)

st.subheader(
    "📄 Input Dataset"
)

st.dataframe(
    df.head(),
    use_container_width=True
)


# =====================================================
# PREPROCESSING
# =====================================================

df.columns = df.columns.astype(str)


# -----------------------------------------------------
# Remove Label column
# -----------------------------------------------------

if " Label" in df.columns:

    labels = df[" Label"]

    df.drop(
        " Label",
        axis=1,
        inplace=True
    )


# -----------------------------------------------------
# Handle invalid values
# -----------------------------------------------------

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.fillna(
    0,
    inplace=True
)


# -----------------------------------------------------
# Match exact training features
# -----------------------------------------------------

training_features = scaler.feature_names_in_

missing_features = [
    feature
    for feature in training_features
    if feature not in df.columns
]


if missing_features:

    st.error(
        f"❌ Dataset is missing "
        f"{len(missing_features)} training features."
    )

    st.write(
        "Missing features:",
        missing_features
    )

    st.stop()


df = df[
    training_features
]


st.success(
    f"✅ {df.shape[1]} Features Matched"
)


# =====================================================
# SCALING
# =====================================================

X_scaled = scaler.transform(
    df
)


# =====================================================
# SEQUENCE GENERATION
# =====================================================

SEQUENCE_LENGTH = 30

X_sequence = create_sequences(
    X_scaled,
    SEQUENCE_LENGTH
)
st.info(f"Rows Loaded: {len(df)}")
st.info(f"Sequences Created: {len(X_sequence)}")

if len(X_sequence) == 0:

    st.error(
        "❌ Dataset must contain at least "
        "30 rows for sequence analysis."
    )

    st.stop()

# =====================================================
# AI PREDICTION (BATCH PROCESSING)
# =====================================================

BATCH_SIZE = 1000

progress_bar = st.progress(0)
status_text = st.empty()

predictions_list = []

total_batches = (len(X_sequence) + BATCH_SIZE - 1) // BATCH_SIZE

with st.spinner("🤖 AI is analysing the uploaded traffic..."):

    for batch_no, start in enumerate(range(0, len(X_sequence), BATCH_SIZE)):

        end = start + BATCH_SIZE

        batch = X_sequence[start:end]

        batch_prediction = model.predict(
            batch,
            batch_size=32,
            verbose=0
        )

        predictions_list.append(batch_prediction)

        progress = (batch_no + 1) / total_batches

        progress_bar.progress(progress)

        status_text.info(
            f"Processing Batch {batch_no + 1}/{total_batches}"
        )

X_pred = np.concatenate(
    predictions_list,
    axis=0
)

progress_bar.empty()
status_text.empty()

st.success("✅ AI Analysis Completed")

# =====================================================
# RECONSTRUCTION ERROR
# =====================================================

reconstruction_error = np.mean(
    np.square(
        X_sequence - X_pred
    ),
    axis=(1, 2)
)


# =====================================================
# ATTACK DETECTION
# =====================================================

predictions = np.where(
    reconstruction_error > threshold,
    "ATTACK",
    "BENIGN"
)


# =====================================================
# RESULT DATAFRAME
# =====================================================

result_df = pd.DataFrame(
    {
        "Sample_ID":
            range(len(predictions)),

        "Anomaly_Score":
            reconstruction_error,

        "Prediction":
            predictions
    }
)


# =====================================================
# SOC DASHBOARD
# =====================================================

st.markdown("---")

st.header(
    "📡 Security Operations Dashboard"
)


# =====================================================
# METRICS
# =====================================================

total_samples = len(
    predictions
)

attack_count = np.sum(
    predictions == "ATTACK"
)

benign_count = np.sum(
    predictions == "BENIGN"
)

attack_percentage = (
    attack_count /
    total_samples
) * 100


max_anomaly = np.max(
    reconstruction_error
)

avg_anomaly = np.mean(
    reconstruction_error
)


# =====================================================
# KPI CARDS
# =====================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="cyber-card">

        <h3>📡 Traffic</h3>

        <h1>{total_samples}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="cyber-card">

        <h3>🚨 Threats</h3>

        <h1>{attack_count}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="cyber-card">

        <h3>🟢 Safe</h3>

        <h1>{benign_count}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="cyber-card">

        <h3>⚠️ Risk</h3>

        <h1>{attack_percentage:.2f}%</h1>

        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# SECURITY STATUS
# =====================================================

st.markdown("---")


if attack_percentage < 10:

    st.markdown(
        """
        <div class="online">

        🟢 SECURITY STATUS : NORMAL

        </div>
        """,
        unsafe_allow_html=True
    )

elif attack_percentage < 40:

    st.warning(
        "🟡 SECURITY STATUS : "
        "SUSPICIOUS ACTIVITY DETECTED"
    )

else:

    st.markdown(
        """
        <div class="danger">

        🔴 SECURITY STATUS : HIGH THREAT DETECTED

        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# TRAFFIC DISTRIBUTION
# =====================================================

st.subheader(
    "🛡️ Traffic Classification"
)


count_df = pd.DataFrame(
    predictions,
    columns=["Prediction"]
)


count_df = (
    count_df
    .value_counts()
    .reset_index()
)


count_df.columns = [
    "Prediction",
    "Count"
]


fig1 = px.pie(
    count_df,
    names="Prediction",
    values="Count",
    hole=0.45
)


fig1.update_layout(
    template="plotly_dark",
    paper_bgcolor="#020617",
    plot_bgcolor="#020617"
)


st.plotly_chart(
    fig1,
    use_container_width=True
)


# =====================================================
# ANOMALY MONITORING GRAPH
# =====================================================

st.subheader(
    "📈 Live Anomaly Monitoring"
)


error_df = pd.DataFrame(
    {
        "Sample":
            range(len(reconstruction_error)),

        "Score":
            reconstruction_error,

        "Status":
            predictions
    }
)


fig2 = px.line(
    error_df,
    x="Sample",
    y="Score",
    color="Status"
)


fig2.add_hline(
    y=float(threshold),
    line_dash="dash",
    annotation_text="Threshold"
)


fig2.update_layout(
    template="plotly_dark",
    paper_bgcolor="#020617",
    plot_bgcolor="#020617"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =====================================================
# SCORE DISTRIBUTION
# =====================================================

st.subheader(
    "📊 Anomaly Score Distribution"
)


fig3 = px.histogram(
    error_df,
    x="Score",
    color="Status",
    nbins=60
)


fig3.add_vline(
    x=float(threshold),
    line_dash="dash",
    annotation_text="Threshold"
)


fig3.update_layout(
    template="plotly_dark",
    paper_bgcolor="#020617",
    plot_bgcolor="#020617"
)


st.plotly_chart(
    fig3,
    use_container_width=True
)


# =====================================================
# TOP THREATS
# =====================================================

st.subheader(
    "🚨 Highest Risk Events"
)


top_threats = (
    result_df
    .sort_values(
        "Anomaly_Score",
        ascending=False
    )
    .head(25)
)


st.dataframe(
    top_threats,
    use_container_width=True
)


# =====================================================
# DOWNLOAD REPORT
# =====================================================

st.subheader(
    "📥 Export Security Report"
)


csv = result_df.to_csv(
    index=False
)


st.download_button(
    "⬇️ Download Threat Report",
    data=csv,
    file_name="AI_Cyber_Threat_Report.csv",
    mime="text/csv"
)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    """
    AI Threat Monitoring Center |
    LSTM Autoencoder |
    CICIDS2017 Network Intrusion Detection
    """
)