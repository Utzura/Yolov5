import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os

# ---- CONFIGURACIÓN DE PÁGINA ----
st.set_page_config(
    page_title="🔍 Detección de Objetos en Tiempo Real",
    page_icon="🎥",
    layout="wide"
)

# ---- ESTILOS PERSONALIZADOS ----
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #1C1F26;
}
.stButton>button {
    border-radius: 10px;
    background-color: #2D3748;
    color: white;
    border: 1px solid #555;
}
.stButton>button:hover {
    background-color: #4A5568;
}
.stCameraInput label {
    color: #E2E8F0 !important;
}
.stDataFrame {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---- SWITCH DE TEMA ----
with st.sidebar:
    tema = st.radio("🎨 Tema de Interfaz", ["🌙 Oscuro", "☀️ Claro"])

if tema == "☀️ Claro":
    st.markdown("""
    <style>
    body { background-color: #F8F9FA; color: black; }
    [data-testid="stSidebar"] { background-color: #F1F3F6; color: black; }
    .stButton>button { background-color: #E2E8F0; color: black; border: 1px solid #CCC; }
    .stButton>button:hover { background-color: #CBD5E0; }
    </style>
    """, unsafe_allow_html=True)

# ---- CARGA DE MODELO ----
@st.cache_resource
def load_model():
    try:
        import yolov5
        model = yolov5.load('yolov5s.pt')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

st.title("🎥 Detección de Objetos con YOLOv5")
st.caption("Captura una imagen y detecta objetos automáticamente usando visión por computadora.")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Configuración")
    conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    filtro = st.radio("🧩 Filtro de imagen", ["Normal", "Inverso (negativo)"])

# ---- CARGAR MODELO ----
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    model.conf = conf
    model.iou = iou

    img = st.camera_input("📸 Toma una foto")

    if img:
        bytes_data = img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if filtro == "Inverso (negativo)":
            cv2_img = cv2.bitwise_not(cv2_img)

        with st.spinner("Detectando objetos..."):
            results = model(cv2_img)

        st.success("✅ Detección completada")

        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.image(results.render()[0], caption="Resultado de detección", use_container_width=True)

        with col2:
            st.subheader("📊 Resumen de objetos detectados")

            predictions = results.pred[0]
            if len(predictions) > 0:
                categories = predictions[:, 5].cpu().numpy().astype(int)
                scores = predictions[:, 4].cpu().numpy()
                labels = [model.names[c] for c in categories]

                df = pd.DataFrame({
                    "Categoría": labels,
                    "Confianza": [f"{s:.2f}" for s in scores]
                })

                st.dataframe(df, use_container_width=True)
                st.bar_chart(df['Categoría'].value_counts())
            else:
                st.info("No se detectaron objetos.")
else:
    st.warning("⚠️ No se pudo cargar el modelo YOLOv5 correctamente.")
