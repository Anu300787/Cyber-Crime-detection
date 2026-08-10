# ============================================================
# Cyber Attack Detection Dashboard
# Time Series Anomaly Detection using LSTM Autoencoder
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Title
# ============================================================

st.title("🛡️ Cyber Attack Detection Dashboard")

st.markdown(
    """
    ### Time Series Anomaly Detection using LSTM Autoencoder

    Upload a network traffic CSV file and the trained LSTM
    Autoencoder will identify anomalous network behaviour.
    """
)


# ============================================================
# Load Model Files
# ============================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        "models/lstm_autoencoder_final.keras"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    threshold = joblib.load(
        "models/threshold.pkl"
    )

    return model, scaler, threshold


# ============================================================
# Load Artifacts
# ============================================================

with st.spinner("Loading AI Model..."):

    model, scaler, threshold = load_artifacts()


# Make sure threshold is a scalar
threshold = float(np.asarray(threshold).squeeze())


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("---")

st.sidebar.write("**Model**")
st.sidebar.info("LSTM Autoencoder")

st.sidebar.write("**Training Data**")
st.sidebar.success("BENIGN Traffic")

st.sidebar.write("**Detection Method**")
st.sidebar.info("Reconstruction Error")

st.sidebar.write("**Sequence Length**")
st.sidebar.success("30")

st.sidebar.write("**Features**")
st.sidebar.success(str(scaler.n_features_in_))

st.sidebar.write("**Threshold**")
st.sidebar.warning(f"{threshold:.4f}")

st.sidebar.write("---")

st.sidebar.caption(
    "Developed using TensorFlow + Streamlit"
)


# ============================================================
# Constants
# ============================================================

SEQUENCE_LENGTH = 30

# Number of sequences processed at one time
INFERENCE_BATCH_SIZE = 256


# ============================================================
# Memory Efficient Sequence Creation
# ============================================================

def create_sequence_batch(
    data,
    start,
    batch_size,
    sequence_length
):

    total_sequences = len(data) - sequence_length + 1

    end = min(
        start + batch_size,
        total_sequences
    )

    if start >= end:
        return np.empty(
            (0, sequence_length, data.shape[1]),
            dtype=np.float32
        )

    sequences = np.stack(
        [
            data[i:i + sequence_length]
            for i in range(start, end)
        ]
    )

    return sequences.astype(np.float32)


# ============================================================
# Hero Metrics
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="Model",
        value="LSTM AE"
    )

with col2:

    st.metric(
        label="Training",
        value="BENIGN"
    )

with col3:

    st.metric(
        label="Sequence Length",
        value=SEQUENCE_LENGTH
    )

with col4:

    st.metric(
        label="Threshold",
        value=f"{threshold:.4f}"
    )


st.write("---")


# ============================================================
# Upload Dataset
# ============================================================

st.markdown(
    """
    ### 📂 Upload Network Traffic Dataset

    Upload a CSV file containing network flow records.
    The application will clean the data, scale the features,
    generate time-series batches and detect anomalous traffic.
    """
)


uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)


# ============================================================
# Main Application
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # Read Dataset
    # ========================================================

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(
            f"Unable to read the uploaded CSV: {e}"
        )

        st.stop()


    st.success(
        "Dataset uploaded successfully!"
    )


    # ========================================================
    # Dataset Preview
    # ========================================================

    with st.expander(
        "Dataset Preview",
        expanded=False
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


    # ========================================================
    # Dataset Summary
    # ========================================================

    st.markdown("## 📊 Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )


    # ========================================================
    # Detect Label Column
    # ========================================================

    label_column = None

    if " Label" in df.columns:

        label_column = " Label"

    elif "Label" in df.columns:

        label_column = "Label"


    if label_column is not None:

        c3.metric(
            "Classes",
            df[label_column].nunique()
        )

    else:

        c3.metric(
            "Classes",
            "Unknown"
        )


    st.write("---")


    # ========================================================
    # Feature Preparation
    # ========================================================

    if label_column is not None:

        actual_labels = df[label_column].copy()

        X = df.drop(
            label_column,
            axis=1
        ).copy()

    else:

        actual_labels = None

        X = df.copy()


    # ========================================================
    # Remove Non-Numeric Columns
    # ========================================================

    # The scaler was trained only on numerical features.
    # This prevents errors caused by string/object columns.

    non_numeric_columns = X.select_dtypes(
        exclude=[np.number]
    ).columns


    if len(non_numeric_columns) > 0:

        X = X.drop(
            columns=non_numeric_columns
        )


    # ========================================================
    # Feature Validation
    # ========================================================

    expected_features = scaler.n_features_in_

    if X.shape[1] != expected_features:

        st.error(
            f"""
            ❌ Incorrect number of features.

            Expected: **{expected_features}**

            Found: **{X.shape[1]}**

            The uploaded CSV must contain the same numerical
            feature structure used when training the model.
            """
        )

        st.stop()


    # ========================================================
    # Data Cleaning
    # ========================================================

    with st.spinner(
        "Cleaning uploaded data..."
    ):

        # Replace infinity values
        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Convert remaining values to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Remove rows containing NaN
        valid_rows = X.notna().all(axis=1)

        X = X.loc[
            valid_rows
        ].reset_index(drop=True)


    # Keep labels aligned with cleaned X
    if actual_labels is not None:

        actual_labels = actual_labels.loc[
            valid_rows
        ].reset_index(drop=True)


    if len(X) == 0:

        st.error(
            "No valid rows remain after data cleaning."
        )

        st.stop()


    st.success(
        f"Data cleaning completed. "
        f"{len(X):,} valid rows available."
    )


    # ========================================================
    # Scaling
    # ========================================================

    with st.spinner(
        "Scaling Features..."
    ):

        try:

            X_scaled = scaler.transform(X)

        except Exception as e:

            st.error(
                f"Scaling failed: {e}"
            )

            st.stop()


    # Convert to float32 to reduce memory usage

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float32
    )


    st.success(
        "Feature scaling completed."
    )


    # ========================================================
    # Number of Sequences
    # ========================================================

    total_sequences = (
        len(X_scaled)
        - SEQUENCE_LENGTH
        + 1
    )


    if total_sequences <= 0:

        st.error(
            f"""
            ❌ Not enough rows for sequence creation.

            Required at least:
            **{SEQUENCE_LENGTH} rows**

            Found:
            **{len(X_scaled)} rows**
            """
        )

        st.stop()


    # ========================================================
    # Sequence Information
    # ========================================================

    st.markdown(
        "## 🔄 Sequence Information"
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Sequences",
        f"{total_sequences:,}"
    )

    a2.metric(
        "Sequence Length",
        SEQUENCE_LENGTH
    )

    a3.metric(
        "Features",
        X_scaled.shape[1]
    )


    st.write("---")

    st.success(
        "Dataset is ready for prediction."
    )


    # ========================================================
    # Prediction
    # ========================================================

    st.markdown(
        "## 🤖 AI Inference"
    )


    if st.button(
        "🚀 Detect Cyber Attacks",
        use_container_width=True
    ):

        # ====================================================
        # Memory Efficient Prediction
        # ====================================================

        reconstruction_errors = []


        progress_bar = st.progress(
            0
        )


        status_text = st.empty()


        with st.spinner(
            "Running LSTM Autoencoder..."
        ):

            for start in range(
                0,
                total_sequences,
                INFERENCE_BATCH_SIZE
            ):

                # --------------------------------------------
                # Create only one batch of sequences
                # --------------------------------------------

                X_batch = create_sequence_batch(
                    X_scaled,
                    start,
                    INFERENCE_BATCH_SIZE,
                    SEQUENCE_LENGTH
                )


                # --------------------------------------------
                # Model Prediction
                # --------------------------------------------

                X_pred = model.predict(
                    X_batch,
                    batch_size=INFERENCE_BATCH_SIZE,
                    verbose=0
                )


                # --------------------------------------------
                # Reconstruction Error
                # --------------------------------------------

                batch_error = np.mean(
                    np.square(
                        X_batch - X_pred
                    ),
                    axis=(1, 2)
                )


                reconstruction_errors.extend(
                    batch_error.tolist()
                )


                # --------------------------------------------
                # Free Memory
                # --------------------------------------------

                del X_batch
                del X_pred


                # --------------------------------------------
                # Progress
                # --------------------------------------------

                processed = min(
                    start + INFERENCE_BATCH_SIZE,
                    total_sequences
                )

                progress = (
                    processed
                    / total_sequences
                )

                progress_bar.progress(
                    progress
                )

                status_text.text(
                    f"Processed "
                    f"{processed:,} / "
                    f"{total_sequences:,} sequences"
                )


        progress_bar.empty()
        status_text.empty()


        # ====================================================
        # Convert Errors to NumPy Array
        # ====================================================

        reconstruction_error = np.asarray(
            reconstruction_errors,
            dtype=np.float32
        )


        # ====================================================
        # Predictions
        # ====================================================

        predictions = np.where(
            reconstruction_error > threshold,
            "ATTACK",
            "BENIGN"
        )


        # ====================================================
        # Results DataFrame
        # ====================================================

        results = pd.DataFrame({

            "Sequence": np.arange(
                1,
                len(predictions) + 1
            ),

            "Reconstruction Error":
                reconstruction_error,

            "Prediction":
                predictions

        })


        # ====================================================
        # Actual Labels
        # ====================================================

        if actual_labels is not None:

            # A sequence ending at position 29 corresponds
            # to the 30th original row.

            sequence_labels = actual_labels.iloc[
                SEQUENCE_LENGTH - 1:
            ].reset_index(drop=True)


            # Make sure lengths match

            sequence_labels = sequence_labels.iloc[
                :len(results)
            ]


            results["Actual Label"] = (
                sequence_labels.values
            )


        st.success(
            "Prediction completed successfully!"
        )


        st.write("---")


        # ====================================================
        # KPI Cards
        # ====================================================

        total_sequences_result = len(
            results
        )


        attack_count = (
            results["Prediction"]
            == "ATTACK"
        ).sum()


        benign_count = (
            results["Prediction"]
            == "BENIGN"
        ).sum()


        attack_percentage = (
            attack_count
            / total_sequences_result
        ) * 100


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Total Sequences",
            f"{total_sequences_result:,}"
        )


        c2.metric(
            "Detected Attacks",
            f"{attack_count:,}"
        )


        c3.metric(
            "Normal Traffic",
            f"{benign_count:,}"
        )


        c4.metric(
            "Attack %",
            f"{attack_percentage:.2f}%"
        )


        st.write("---")


        # ====================================================
        # Results Preview
        # ====================================================

        st.markdown(
            "## 📋 Prediction Results"
        )


        st.dataframe(
            results.head(100),
            use_container_width=True
        )


        st.write("---")


        # ====================================================
        # High Risk Alert
        # ====================================================

        if attack_percentage > 30:

            st.error(
                "⚠️ High anomaly rate detected "
                "in the uploaded network traffic."
            )

        elif attack_percentage > 10:

            st.warning(
                "⚠️ Moderate anomaly activity detected."
            )

        else:

            st.success(
                "✅ Network traffic appears mostly normal."
            )


        # ====================================================
        # Dashboard
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## 📊 Cyber Security Dashboard"
        )


        # ====================================================
        # Charts
        # ====================================================

        left, right = st.columns(2)


        # ====================================================
        # Pie Chart
        # ====================================================

        with left:

            prediction_counts = (
                results["Prediction"]
                .value_counts()
            )


            fig = px.pie(
                values=prediction_counts.values,
                names=prediction_counts.index,
                title="Traffic Distribution",
                hole=0.55
            )


            fig.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ====================================================
        # Reconstruction Error Histogram
        # ====================================================

        with right:

            fig = px.histogram(
                results,
                x="Reconstruction Error",
                color="Prediction",
                nbins=60,
                title="Reconstruction Error Distribution"
            )


            fig.add_vline(
                x=float(threshold),
                line_dash="dash",
                line_width=3,
                annotation_text="Threshold"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ====================================================
        # Reconstruction Error Trend
        # ====================================================

        st.markdown(
            "### 📈 Reconstruction Error Trend"
        )


        fig = px.line(
            results,
            x="Sequence",
            y="Reconstruction Error",
            title="Reconstruction Error Across Sequences"
        )


        fig.add_hline(
            y=float(threshold),
            line_dash="dash",
            annotation_text="Threshold"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # Box Plot
        # ====================================================

        st.markdown(
            "### 📦 Error Comparison"
        )


        fig = px.box(
            results,
            x="Prediction",
            y="Reconstruction Error",
            color="Prediction",
            title="Reconstruction Error by Prediction"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # Highest Reconstruction Errors
        # ====================================================

        st.markdown(
            "### 🚨 Top Suspicious Sequences"
        )


        top_errors = (
            results
            .sort_values(
                by="Reconstruction Error",
                ascending=False
            )
            .head(20)
        )


        st.dataframe(
            top_errors,
            use_container_width=True
        )


        # ====================================================
        # Download Results
        # ====================================================

        csv = results.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ====================================================
        # Security Intelligence Layer
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## 🛡️ Security Intelligence Report"
        )


        # ====================================================
        # Security Statistics
        # ====================================================

        total_sequences_result = len(
            results
        )


        attack_count = (
            results["Prediction"]
            == "ATTACK"
        ).sum()


        normal_count = (
            results["Prediction"]
            == "BENIGN"
        ).sum()


        attack_percentage = (
            attack_count
            / total_sequences_result
        ) * 100


        # ====================================================
        # Security Score
        # ====================================================

        security_score = (
            100 - attack_percentage
        )


        security_score = max(
            0,
            round(
                security_score,
                2
            )
        )


        # ====================================================
        # Risk Level
        # ====================================================

        if security_score >= 90:

            risk_level = "LOW"
            risk_icon = "🟢"

        elif security_score >= 70:

            risk_level = "MEDIUM"
            risk_icon = "🟡"

        elif security_score >= 40:

            risk_level = "HIGH"
            risk_icon = "🟠"

        else:

            risk_level = "CRITICAL"
            risk_icon = "🔴"


        # ====================================================
        # Security Metrics
        # ====================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Security Score",
                f"{security_score}/100"
            )


        with col2:

            st.metric(
                "Attack Probability",
                f"{attack_percentage:.2f}%"
            )


        with col3:

            st.metric(
                "Risk Level",
                f"{risk_icon} {risk_level}"
            )


        # ====================================================
        # Security Alert
        # ====================================================

        if risk_level in [
            "HIGH",
            "CRITICAL"
        ]:

            st.error(
                f"""
                🚨 Security Alert

                Suspicious network behaviour detected.

                Risk Level: {risk_level}

                Immediate investigation recommended.
                """
            )


        elif risk_level == "MEDIUM":

            st.warning(
                """
                ⚠️ Moderate Threat Activity Detected

                Continuous monitoring recommended.
                """
            )


        else:

            st.success(
                """
                ✅ Network Status Normal

                No significant anomaly detected.
                """
            )


        # ====================================================
        # Executive Summary
        # ====================================================

        st.markdown(
            "### 📑 Executive Summary"
        )


        summary = f"""
        The AI-based intrusion detection system analyzed
        **{total_sequences_result:,} network sequences**.

        **Normal Traffic:**  
        🟢 {normal_count:,} sequences

        **Suspicious Traffic:**  
        🔴 {attack_count:,} sequences

        **Attack Probability:**  
        {attack_percentage:.2f}%

        **Current Security Risk:**  
        {risk_level}

        The detection decision is based on:

        - LSTM Autoencoder reconstruction error
        - Learned normal network behaviour
        - Anomaly threshold
        """


        st.info(summary)