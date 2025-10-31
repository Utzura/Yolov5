import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="🔍 Detección de Objetos con YOLOv5",
    page_icon="🎥",
    layout="wide",
)

# --- ESTILO OSCURO ELEGANTE ---
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ffffff !important;
    }
    .css-1d391kg, .css-18e3th9 {
        background-color: #161b22 !important;
    }
    .stButton>button {
        background-color: #1f6feb !important;
        color: white !important;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #388bfd !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.title("🎥 Detección de Objetos en Tiempo Real con YOLOv5")
st.markdown("Detecta objetos en imágenes o desde tu cámara usando el modelo **YOLOv5 (Ultralytics)**.")

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov5s.pt")  # Se descarga automáticamente si no existe
        return model
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        return None

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model is None:
    st.stop()

st.sidebar.header("⚙️ Parámetros de detección")
conf_thres = st.sidebar.slider("Confianza mínima", 0.1, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("Umbral IoU", 0.1, 1.0, 0.45, 0.05)
st.sidebar.markdown("---")

# --- CÁMARA / SUBIDA ---
option = st.radio("📸 Elige una fuente de imagen:", ["Capturar con cámara", "Subir imagen"])
image = None

if option == "Capturar con cámara":
    picture = st.camera_input("Captura una imagen para analizar")
    if picture:
        image = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), cv2.IMREAD_COLOR)
else:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)

# --- PROCESAR IMAGEN ---
if image is not None:
    st.markdown("### 🔍 Resultados de detección")
    with st.spinner("Analizando imagen..."):
        results = model.predict(image, conf=conf_thres, iou=iou_thres)
    
    # Dibujar detecciones sobre la imagen
    annotated_img = results[0].plot()
    detections = results[0].boxes.data.cpu().numpy()

    # Mostrar resultados
    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Imagen con detecciones", use_container_width=True)

    with col2:
        if len(detections) > 0:
            st.subheader("📋 Objetos detectados")
            names = model.names
            data = []
            for *box, conf, cls in detections:
                data.append({
                    "Categoría": names[int(cls)],
                    "Confianza": round(float(conf), 2)
                })
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df["Categoría"].value_counts())
        else:
            st.info("No se detectaron objetos con los parámetros actuales.")

st.markdown("---")
st.caption("Aplicación creada con ❤️ usando Streamlit + Ultralytics YOLOv5.")
