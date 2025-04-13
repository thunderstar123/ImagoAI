import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.express as px
import plotly.graph_objects as go

# Data Loading and Preprocessing
@st.cache_data
def load_data():
    data = pd.read_csv('TASK-ML-INTERN.csv')
    X = data.drop(['hsi_id', 'vomitoxin_ppb'], axis=1)
    y = data['vomitoxin_ppb']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, data

# Model Building
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Visualization Functions
def plot_spectral_bands(data, X):
    fig = plt.figure(figsize=(15, 6))
    for i in range(len(X)):
        plt.plot(range(X.shape[1]), X[i], alpha=0.1)
    plt.title('Spectral Bands for All Samples')
    plt.xlabel('Wavelength Band')
    plt.ylabel('Reflectance')
    return fig

def plot_heatmap(data, X):
    fig = plt.figure(figsize=(15, 8))
    sns.heatmap(X[:10], cmap='viridis')
    plt.title('Heatmap of Spectral Data (First 10 Samples)')
    plt.xlabel('Wavelength Band')
    plt.ylabel('Sample')
    return fig

def main():
    st.set_page_config(page_title="Corn Mycotoxin Analysis", layout="wide")
    st.title("Corn Mycotoxin Analysis and Prediction")
    
    # Load and preprocess data
    X_scaled, y, scaler, data = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Predictions"])
    
    if page == "Data Exploration":
        st.header("Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(data.head())
            
            st.subheader("Basic Statistics")
            st.write(data.describe())
        
        with col2:
            st.subheader("Spectral Bands Visualization")
            st.pyplot(plot_spectral_bands(data, X_scaled))
            
            st.subheader("Heatmap Visualization")
            st.pyplot(plot_heatmap(data, X_scaled))
        
        # Dimensionality Reduction
        st.header("Dimensionality Reduction")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("PCA Analysis")
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            fig_pca = px.scatter_3d(
                None, x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                color=y,
                title="PCA Visualization"
            )
            st.plotly_chart(fig_pca)
        
        with col4:
            st.subheader("t-SNE Analysis")
            X_tsne = TSNE(n_components=2).fit_transform(X_scaled)
            
            fig_tsne = px.scatter(
                None, x=X_tsne[:, 0], y=X_tsne[:, 1],
                color=y,
                title="t-SNE Visualization"
            )
            st.plotly_chart(fig_tsne)
    
    elif page == "Model Training":
        st.header("Model Training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Model parameters
        epochs = st.slider("Number of epochs", 10, 200, 100)
        batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = build_model(X_train.shape[1])
                
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Save model
                model.save('model.h5')
                
                # Evaluate model
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                st.success("Model training completed!")
                
                col5, col6, col7 = st.columns(3)
                col5.metric("MAE", f"{mae:.2f}")
                col6.metric("RMSE", f"{rmse:.2f}")
                col7.metric("R² Score", f"{r2:.2f}")
                
                # Plot training history
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig_history.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig_history)
                
                # Plot predictions
                fig_pred = px.scatter(
                    x=y_test, y=y_pred.flatten(),
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title='Actual vs Predicted Values'
                )
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                            y=[y_test.min(), y_test.max()],
                                            mode='lines', name='Perfect Prediction'))
                st.plotly_chart(fig_pred)
    
    else:  # Predictions page
        st.header("Make Predictions")
        
        uploaded_file = st.file_uploader("Upload new data (CSV format)", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load and preprocess new data
                new_data = pd.read_csv(uploaded_file)
                X_new = new_data.drop(['hsi_id', 'vomitoxin_ppb'], axis=1, errors='ignore')
                X_new_scaled = scaler.transform(X_new)
                
                # Load model and make predictions
                model = tf.keras.models.load_model('model.h5')
                predictions = model.predict(X_new_scaled)
                
                # Display results
                results_df = pd.DataFrame({
                    'Sample': range(len(predictions)),
                    'Predicted DON (ppb)': predictions.flatten()
                })
                
                st.subheader("Prediction Results")
                st.write(results_df)
                
                # Plot predictions
                fig_predictions = px.line(
                    results_df, x='Sample', y='Predicted DON (ppb)',
                    title='Predicted DON Concentrations'
                )
                st.plotly_chart(fig_predictions)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()