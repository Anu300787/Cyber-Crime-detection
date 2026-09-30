🛡️ AI-Powered Cyber Attack Detection

An AI-powered network intrusion detection system that uses an LSTM Autoencoder to learn patterns from benign network traffic and detect potential DDoS attacks through anomaly detection.

The project is built using the CICIDS2017 dataset and provides a Streamlit-based interface for detecting anomalous network traffic.

---

🚀 Features

- Detects anomalous network traffic using Deep Learning
- Uses an LSTM Autoencoder for sequence-based anomaly detection
- Trained primarily on BENIGN traffic
- Detects potential DDoS attacks based on reconstruction error
- Uses a configurable anomaly detection threshold
- Data preprocessing and feature scaling
- Streamlit interface for easy interaction
- Saves trained model, scaler, and threshold for inference

---

🧠 How It Works

The system follows an anomaly-detection approach rather than directly training a classifier on attack labels.

Workflow

CICIDS2017 Network Traffic
          ↓
     Data Cleaning
          ↓
    Feature Selection
          ↓
     Feature Scaling
          ↓
  Create Sequences
     (Length = 30)
          ↓
   LSTM Autoencoder
          ↓
 Reconstruct Input Sequence
          ↓
 Calculate Reconstruction Error
          ↓
 Compare with Threshold
       ↙        ↘
   Normal       Anomaly
                ↓
          Possible Attack

The model is trained using benign traffic, allowing it to learn the characteristics of normal network behavior.

During detection, the input traffic is reconstructed by the autoencoder. A high reconstruction error indicates that the traffic differs significantly from the learned normal pattern and may represent an anomaly.

---

🏗️ Model Architecture

The LSTM Autoencoder uses the following architecture:

Input Sequence
     ↓
LSTM (64 units)
     ↓
LSTM (32 units)
     ↓
RepeatVector
     ↓
LSTM (32 units)
     ↓
LSTM (64 units)
     ↓
TimeDistributed Dense (68)
     ↓
Reconstructed Sequence

Model Details

- Input features: 68
- Sequence length: 30
- Encoder: LSTM layers
- Decoder: LSTM layers
- Output: Reconstructed input sequence
- Total parameters: ~84K

---

📊 Dataset

This project uses the CICIDS2017 dataset, a widely used dataset for network intrusion detection research.

The project focuses on network traffic containing:

- BENIGN traffic
- DDoS traffic

The processed dataset contains approximately:

- 223,082 network flows
- 95,068 BENIGN flows
- 128,014 DDoS flows
- 68 input features

The model is trained on benign traffic to learn normal network behavior.

---

🛠️ Technologies Used

- Python
- TensorFlow / Keras
- LSTM Autoencoder
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- CICIDS2017

---
📁 Project Structure

Cyber-Attack-Detection/
│
├── app.py
├── lstm_autoencoder_final.h5
├── scaler.pkl
├── threshold.pkl
├── requirements.txt
├── README.md
│
└── data/
    └── cleaned_network_flows.csv

«Note: The dataset may not be included in the repository because of its size. Download and preprocess the CICIDS2017 dataset separately if required.»

---

⚙️ Installation


1. Clone the repository

git clone https://github.com/Anu300787/Cyber-Attack-Detection.git

2. Create a virtual environment

python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

---

▶️ Run the Application

Start the Streamlit application using:

streamlit run app.py

The application will open in your browser.

---

🔍 Detection Logic

The model calculates the reconstruction error between the original network sequence and the sequence reconstructed by the autoencoder.

Reconstruction Error > Threshold
            ↓
        Anomaly
            ↓
   Potential Cyber Attack

Otherwise:

Reconstruction Error ≤ Threshold
            ↓
         Normal

The threshold used by the application is stored in:

threshold.pkl

---

💾 Saved Model Files

File| Purpose
"lstm_autoencoder_final.h5"| Trained LSTM Autoencoder
"scaler.pkl"| Feature scaling object
"threshold.pkl"| Anomaly detection threshold
"app.py"| Streamlit application

---

🔐 Why Anomaly Detection?

Traditional supervised intrusion detection systems require examples of different attack types during training.

This project instead uses an autoencoder-based anomaly detection approach:

1. Train the model on normal/benign traffic.
2. Learn the underlying patterns of normal network behavior.
3. Evaluate new traffic based on reconstruction error.
4. Flag traffic that significantly differs from normal behavior.

This approach can be useful when labeled examples of every possible attack are not available.

---

📌 Project Highlights

- Implemented an LSTM Autoencoder from scratch using TensorFlow/Keras
- Processed real-world network traffic from CICIDS2017
- Used sequential modeling for network-flow anomaly detection
- Developed an anomaly detection mechanism using reconstruction error
- Built a Streamlit application for practical inference
- Created a reusable pipeline using saved model, scaler, and threshold files

---

🔮 Future Improvements

- Extend detection to additional attack categories
- Experiment with GRU and Transformer-based architectures
- Improve feature selection and preprocessing
- Add real-time network traffic monitoring
- Deploy the application as a cloud-based detection service
- Add detailed attack analytics and visualization
- Evaluate the model using precision, recall, F1-score, ROC-AUC and confusion matrix

---

👩‍💻 Author

Anushka Tripathi

B.Tech CSE (Data Science)
Interested in Artificial Intelligence, Machine Learning and Cybersecurity

---

⭐ Acknowledgements

- CICIDS2017 dataset by the Canadian Institute for Cybersecurity
- TensorFlow/Keras for deep learning implementation
- Streamlit for the interactive application interface