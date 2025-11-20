import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json
import base64
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------- Utilities: data generation & anomaly injection ----------

def generate_synthetic_data(n_channels=3, length=2000, seed=42):
    np.random.seed(seed)
    t = np.arange(length)
    data = []
    for ch in range(n_channels):
        
        freq = 0.005 + 0.002 * ch
        phase = np.random.rand() * 2 * np.pi
        signal = 0.5 * np.sin(2 * np.pi * freq * t + phase)
        trend = 0.0002 * t * (0.5 + 0.5 * ch)
        noise = 0.05 * np.random.randn(length)
        channel = signal + trend + noise
        data.append(channel)
    data = np.vstack(data).T  
    df = pd.DataFrame(data, columns=[f'ch_{i}' for i in range(n_channels)])
    df['time'] = t
    return df


def inject_label_anomalies(df, anomaly_windows=None, magnitude=2.0, seed=10):
    """
    Inject anomalies into the dataframe in the specified windows.
    anomaly_windows: list of (start, end) index ranges. If None, generate some.
    magnitude: multiplier for deviation
    Returns: (df_injected, labels_df) where labels_df has columns ['time','label']
    """
    rng = np.random.RandomState(seed)
    df2 = df.copy()
    length = len(df)
    if anomaly_windows is None:
        anomaly_windows = []
        
        for i in range(3):
            start = int(length * (0.15 + 0.25 * i))
            end = start + int(length * 0.03)
            anomaly_windows.append((start, min(end, length - 1)))

    labels = np.zeros(length, dtype=int)
    for (s, e) in anomaly_windows:
        labels[s:e] = 1
        
        for col in df2.columns:
            if col == 'time':
                continue
            
            df2.loc[s:e, col] += magnitude * (1 + rng.randn(e - s + 1) * 0.2)
            
            if rng.rand() < 0.5:
                df2.loc[s:e, col] += np.linspace(0, magnitude * 0.5, e - s + 1)
    labels_df = pd.DataFrame({'time': df2['time'].values, 'label': labels})
    return df2, labels_df


# ---------- Preprocessing: sliding windows & scalers ----------
from sklearn.preprocessing import StandardScaler


def create_windows(data, window_size=50, step=1):
    """Return 3D array (n_windows, window_size, n_channels) and indices of window centers"""
    n = len(data)
    channels = [c for c in data.columns if c != 'time']
    arr = data[channels].values
    windows = []
    centers = []
    starts = []
    for i in range(0, n - window_size + 1, step):
        windows.append(arr[i:i + window_size])
        centers.append(i + window_size // 2)
        starts.append(i)
    windows = np.array(windows)
    return windows, np.array(centers), np.array(starts), channels


# ---------- Model: LSTM Autoencoder builder ----------

def build_lstm_autoencoder(timesteps, n_features, latent_dim=64):
    inputs = Input(shape=(timesteps, n_features))
    
    x = LSTM(latent_dim, activation='tanh', return_sequences=False)(inputs)
    
    x = RepeatVector(timesteps)(x)
    
    x = LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(n_features))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# ---------- Scoring & thresholding ----------

def reconstruction_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    
    mse = np.mean(np.mean(np.square(X - X_pred), axis=2), axis=1)
    return mse


def dynamic_threshold(errors, method='std', k=3.0, percentile=95):
    if method == 'std':
        mu = np.mean(errors)
        sigma = np.std(errors)
        return mu + k * sigma
    elif method == 'percentile':
        return np.percentile(errors, percentile)
    else:
        raise ValueError('Unknown method')


# ---------- Streamlit UI ----------

st.set_page_config(layout='wide', page_title='LSTM Autoencoder — Anomaly Detection', initial_sidebar_state='expanded')

st.title('Anomaly Detection — LSTM Autoencoder')
st.markdown('This Streamlit app generates synthetic telemetry, injects labelled anomalies, trains an LSTM autoencoder on normal windows and visualizes anomaly scores and detections.')

# Sidebar controls
st.sidebar.header('Data & Simulation')
n_channels = st.sidebar.slider('Number of channels', min_value=1, max_value=8, value=3)
length = st.sidebar.number_input('Time series length (samples)', min_value=500, max_value=20000, value=2000, step=100)
window_size = st.sidebar.slider('Window size (timesteps)', 10, 300, 64)
step = st.sidebar.number_input('Window step', min_value=1, max_value=window_size, value=1)

st.sidebar.header('Anomaly Injection')
num_auto_windows = st.sidebar.slider('Number of injected anomaly windows', 0, 10, 3)
man_mag = st.sidebar.slider('Anomaly magnitude multiplier', 0.5, 5.0, 2.0)

st.sidebar.header('Model & Training')
latent_dim = st.sidebar.slider('Latent dimension (LSTM units)', 8, 256, 64)
epochs = st.sidebar.number_input('Epochs', min_value=1, max_value=200, value=30)
batch_size = st.sidebar.number_input('Batch size', min_value=1, max_value=512, value=128)
threshold_method = st.sidebar.selectbox('Threshold method', ['std', 'percentile'])
threshold_k = st.sidebar.slider('k (for std method)', 0.5, 6.0, 3.0)
threshold_percentile = st.sidebar.slider('Percentile (for percentile method)', 80, 99, 95)

# Buttons
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button('Generate synthetic data'):
        df = generate_synthetic_data(n_channels=n_channels, length=int(length), seed=42)
        st.session_state['df_raw'] = df
        st.success('Synthetic data generated')
with col2:
    if st.button('Inject anomalies (labels)'):
        if st.session_state['df_raw'] is None:
            st.error('Generate data first')
        else:
           
            L = len(st.session_state['df_raw'])
            windows = []
            base = int(L * 0.02)
            for i in range(num_auto_windows):
                start = int(L * (0.1 + i * 0.25))
                end = min(L - 1, start + base + i * 10)
                windows.append((start, end))
            df2, labels = inject_label_anomalies(st.session_state['df_raw'], anomaly_windows=windows, magnitude=man_mag, seed=10)
            st.session_state['df_injected'] = df2
            st.session_state['labels'] = labels
            st.success(f'Injected {len(windows)} anomaly windows')
with col3:
    if st.button('Reset'):
        st.session_state.clear()
        st.experimental_rerun()

# Show a preview of data
st.subheader('Data preview')
if st.session_state.get('df_injected') is not None:
    df_show = st.session_state['df_injected']
elif st.session_state.get('df_raw') is not None:
    df_show = st.session_state['df_raw']
else:
    df_show = None

if df_show is not None:
    st.dataframe(df_show.head(200))
else:
    st.info('Generate synthetic data to see preview')

# Training panel
st.subheader('Train LSTM Autoencoder')
if st.button('Prepare & Train'):
    if st.session_state.get('df_injected') is None and st.session_state.get('df_raw') is None:
        st.error('Generate data (and inject anomalies) first')
    else:
        df_source = st.session_state.get('df_injected') if st.session_state.get('df_injected') is not None else st.session_state.get('df_raw')
        labels_df = st.session_state.get('labels') if st.session_state.get('labels') is not None else pd.DataFrame({'time': df_source['time'].values, 'label': np.zeros(len(df_source), dtype=int)})

       
        windows, centers, starts, channels = create_windows(df_source, window_size=window_size, step=step)
        st.write(f'Prepared {len(windows)} windows — each shape {windows.shape[1:]}')

        
        label_array = labels_df['label'].values
        window_labels = (label_array[centers] > 0).astype(int)

        
        X_train = windows[window_labels == 0]
        st.write(f'Using {len(X_train)} normal windows for training')

        # scaling
        nsamples, nt, nfeat = X_train.shape
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, nfeat)
        scaler.fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d).reshape(nsamples, nt, nfeat)

        # model
        model = build_lstm_autoencoder(nt, nfeat, latent_dim=latent_dim)
        st.write(model.summary())

        
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train_scaled, X_train_scaled, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
        st.success('Training complete')

        # Score all windows
        windows_2d = windows.reshape(-1, nfeat)
        windows_scaled = scaler.transform(windows_2d).reshape(-1, nt, nfeat)
        errors = reconstruction_errors(model, windows_scaled)

        # threshold
        if threshold_method == 'std':
            thresh = dynamic_threshold(errors, method='std', k=threshold_k)
        else:
            thresh = dynamic_threshold(errors, method='percentile', percentile=threshold_percentile)

        preds = (errors > thresh).astype(int)

        
        results_df = pd.DataFrame({'center_idx': centers, 'error': errors, 'pred': preds})
        
        results_df['label'] = label_array[results_df['center_idx']]

        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['results_df'] = results_df
        st.session_state['channels'] = channels
        st.session_state['window_size'] = window_size
        st.session_state['df_source'] = df_source
        st.session_state['thresh'] = float(thresh)
        st.session_state['history'] = history.history

        st.success('Results saved in session state')

# Visualization
st.subheader('Visualization')
if st.session_state.get('results_df') is not None:
    results_df = st.session_state['results_df']
    df_source = st.session_state['df_source']
    channels = st.session_state['channels']

    # Error plot across time
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(results_df['center_idx'], results_df['error'], label='reconstruction error')
    ax.axhline(st.session_state['thresh'], color='r', linestyle='--', label=f'threshold={st.session_state["thresh"]:.4f}')
    ax.set_xlabel('time index')
    ax.set_ylabel('reconstruction error')
    ax.legend()
    st.pyplot(fig)

    # prediction timeline
    fig2, ax2 = plt.subplots(figsize=(12, 2))
    ax2.step(results_df['center_idx'], results_df['pred'], where='mid')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('time index')
    ax2.set_ylabel('anomaly (1/0)')
    st.pyplot(fig2)

    
    show_ch = st.multiselect('Channels to overlay', options=list(channels), default=list(channels)[:2])
    if len(show_ch) > 0:
        fig3, ax3 = plt.subplots(len(show_ch), 1, figsize=(12, 2 * len(show_ch)), sharex=True)
        if len(show_ch) == 1:
            ax3 = [ax3]
        for i, ch in enumerate(show_ch):
            ax3[i].plot(df_source['time'], df_source[ch], label=ch)
            
            for idx, row in results_df.iterrows():
                if row['pred'] == 1:
                    c = int(row['center_idx'])
                    s = max(0, c - st.session_state['window_size'] // 2)
                    e = min(len(df_source) - 1, c + st.session_state['window_size'] // 2)
                    ax3[i].axvspan(s, e, alpha=0.2, color='red')
            ax3[i].legend()
        st.pyplot(fig3)

    # Metrics summary
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = results_df['label'].values
    y_pred = results_df['pred'].values
    if y_true.sum() == 0:
        st.info('No true anomalies present in labels — precision/recall not meaningful')
    else:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        st.metric('Precision', f'{prec:.3f}')
        st.metric('Recall', f'{rec:.3f}')
        st.metric('F1', f'{f1:.3f}')

    # Downloads: results CSV and model save
    buf = BytesIO()
    results_df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button('Download results CSV', data=buf, file_name='anomaly_results.csv', mime='text/csv')

    # Save model to tempfile and allow download
    tmpdir = tempfile.mkdtemp()
    model_path = os.path.join(tmpdir, 'lstm_autoencoder.keras')
    st.session_state['model'].save(model_path)
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    st.download_button('Download trained model (.keras)', data=model_bytes, file_name='lstm_autoencoder.keras', mime='application/octet-stream')

    # Show training curve
    hist = st.session_state.get('history')
    if hist is not None:
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.plot(hist.get('loss', []), label='train loss')
        ax4.set_xlabel('epoch')
        ax4.set_ylabel('loss')
        ax4.legend()
        st.pyplot(fig4)

else:
    st.info('Train model to enable visualizations')




