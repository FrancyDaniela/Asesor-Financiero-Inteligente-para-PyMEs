import uuid
import streamlit as st
from src.agente import ejecutar_agente, extraer_texto

# OJO: ya NO se hace importlib.reload(src.agente). Streamlit re-ejecuta este
# script en cada mensaje; recargar el módulo creaba un MemorySaver nuevo cada
# vez y el agente "olvidaba" la conversación. Si cambias agente.py, reinicia
# `streamlit run app.py`.

st.set_page_config(
    page_title="Asesor Inteligente de Microcréditos PyME",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Asesor Financiero Inteligente para PyMEs")
st.caption("Sistema Neuro-Agéntico de Evaluación Crediticia (Deep Learning + Agente LangGraph)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Un thread_id único por sesión de navegador (antes era fijo para todos).
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.info("Este agente utiliza una **Red Neuronal Deep Learning** para evaluar el riesgo crediticio y herramientas de simulación financiera.")
    if st.button("Limpiar Conversación"):
        st.session_state.messages = []
        # Nuevo hilo: así el agente también olvida, no solo la pantalla.
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu consulta sobre tu PyME o microcrédito..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando la información con el modelo neuronal..."):
            try:
                respuesta = ejecutar_agente(prompt, thread_id=st.session_state.thread_id)
                contenido_respuesta = extraer_texto(respuesta["messages"][-1])
                st.markdown(contenido_respuesta)
                st.session_state.messages.append({"role": "assistant", "content": contenido_respuesta})
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")