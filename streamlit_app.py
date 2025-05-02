import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model dan Aset ---
@st.cache_resource
def load_model_assets():
    model = tf.keras.models.load_model('clickbait_classifier_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('word_index.pkl', 'rb') as f:
        word_index = pickle.load(f)
    return model, label_encoder, word_index

model, label_encoder, word_index = load_model_assets()

# --- Preprocessing Teks ---
def preprocess_input_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def text_to_sequence(tokens, word_index, max_length=100):
    sequence = [word_index[word] for word in tokens if word in word_index]
    padded = pad_sequences([sequence], maxlen=max_length, padding='post')
    return padded

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="Klasifikasi Clickbait", layout="wide")

# --- CSS Kustom ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #121212;
        color: #f0f2f6;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #bbbbbb;
    }
    .info-box {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 10px;
    }
    .result-box {
        background-color: #1f1f1f;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .confidence {
        color: #f9c74f;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .footer {
        font-size: 0.85rem;
        color: #888;
        text-align: center;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Layout Info dan Input ---
col1, col2 = st.columns([0.8, 2])

with col1:
    st.markdown("### ℹ️ Informasi Penting")
    st.markdown("#### Apa yang Dilakukan Aplikasi Ini?")
    st.write("""
    Aplikasi ini memprediksi apakah sebuah judul berita bersifat **Clickbait** atau **Bukan Clickbait**.
    """)

    st.markdown("#### Kriteria Clickbait:")
    st.markdown("""
    1. Judul memancing rasa penasaran tanpa memberi informasi lengkap.
    2. Penggunaan kalimat atau frasa berupa kalimat tanya seperti “apakah kamu tahu?”.  
    3. Mengandung kata-kata hiperbola, dramatis, atau sensasional.  
    4. Bertujuan menarik klik, bukan menyampaikan informasi inti.
    5. Terdapat tanda baca ‘?’ dan “!” atau kaliamat pertanyaan
    """)

    st.markdown("#### Contoh:")
    st.markdown("""
    - 🧨 *Clickbait*:  
      *"Anda tidak akan percaya apa yang terjadi selanjutnya!"*

    - ✅ *Bukan Clickbait*:  
      *"Presiden Resmikan Jalan Tol Baru di Jawa Tengah"*
    """)

with col2:
    st.markdown("<div class='main-title'>📰 Klasifikasi Judul Clickbait</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Pilih metode input: ketik langsung atau unggah file CSV berisi judul berita.</div>", unsafe_allow_html=True)

    input_method = st.radio("Pilih Metode Input", ["📄 Input Manual", "📁 Upload CSV"], horizontal=True)

    if input_method == "📄 Input Manual":
        user_input = st.text_area("Masukkan Judul Berita", height=100, placeholder="Contoh: Anda tidak akan percaya siapa yang datang ke pesta itu...")

        if st.button("🔍 Prediksi Judul", use_container_width=True):
            if user_input.strip() == "":
                st.warning("⚠️ Silakan masukkan judul berita terlebih dahulu.")
            else:
                with st.spinner('⏳ Menganalisis judul...'):
                    time.sleep(1.2)
                    tokens = preprocess_input_text(user_input)
                    sequence = text_to_sequence(tokens, word_index)
                    prediction = model.predict(sequence)[0][0]
                    label = "🧨 Clickbait" if prediction >= 0.5 else "✅ Bukan Clickbait"
                    confidence = prediction if prediction >= 0.5 else 1 - prediction

                st.markdown(f"""
                    <div class="result-box">
                        <h3>Hasil: {label}</h3>
                        <p class="confidence">Probabilitas: {confidence*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

    elif input_method == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Unggah file CSV berisi kolom 'judul'", type=["csv"])

        if uploaded_file:
            try:
                # Coba utf-8 lalu fallback ke ISO-8859-1 jika gagal
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            except Exception as e:
                st.error(f"❌ Gagal membaca file CSV: {e}")
                st.stop()

            if 'judul' not in df.columns:
                st.error("❌ File harus memiliki kolom bernama 'judul'.")
            else:
                with st.spinner("⏳ Memproses dan memprediksi..."):
                    results = []
                    for text in df['judul']:
                        tokens = preprocess_input_text(str(text))
                        sequence = text_to_sequence(tokens, word_index)
                        pred = model.predict(sequence)[0][0]
                        label = "Clickbait" if pred >= 0.5 else "Bukan Clickbait"
                        confidence = round(pred if pred >= 0.5 else 1 - pred, 4)
                        results.append((text, label, confidence))

                    result_df = pd.DataFrame(results, columns=["Judul", "Label", "Probabilitas"])
                    st.success("✅ Prediksi selesai!")
                    st.dataframe(result_df)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Unduh Hasil Prediksi CSV", data=csv, file_name="hasil_prediksi_clickbait.csv", mime="text/csv")

# --- Footer ---
st.markdown("""
    <hr>
    <div class="footer">
        © 2025 Sistem Klasifikasi Judul Berita Clickbait • Dibuat oleh M. Mahdum Al-A'Lam.<br>
        Teknologi: Python, TensorFlow, FastText, BiLSTM, Streamlit.
    </div>
""", unsafe_allow_html=True)
