# 🌽 DON Prediction from Hyperspectral Corn Data

This project uses hyperspectral imaging to predict DON (Deoxynivalenol) levels in corn using a neural network. It includes preprocessing, spectral visualization, PCA/t-SNE analysis, model training, and a Streamlit app for real-time CSV-based predictions.

---

## ⚙️ How to Run

```bash
# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate environment
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Launch Streamlit app
streamlit run app.py


├── app.py                  # Streamlit web app
├── TASK-ML-INTERN.csv      # Hyperspectral dataset
├── model.h5                # Saved model (created after training)
├── requirements.txt        # List of Python packages
├── report.md               # Summary of methodology and findings
└── README.md               # Project overview and instructions


Features
Spectral reflectance visualization (line plots & heatmaps)

PCA & t-SNE dimensionality reduction

Neural network training & evaluation (MAE, RMSE, R²)

Interactive Streamlit app for training and predictions

CSV uploader for new data inference

