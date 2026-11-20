import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.tools import evaluar_riesgo_neuronal_tool, calcular_capacidad_pago_tool

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No se encontró GOOGLE_API_KEY en el archivo .env")
api_key = api_key.strip().strip('"').strip("'")

# El nombre del modelo se lee del .env para no tener que tocar el código
# cada vez que Google retira un modelo (gemini-1.5 y gemini-2.0 ya no existen).
MODELO_LLM = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# max_retries=1: la librería por defecto reintenta varias veces por su cuenta con
# esperas crecientes; sumado a nuestros reintentos, la app podía quedarse minutos
# "pensando". timeout=40: si Google no responde en 40 s, se corta y se pasa al respaldo.
llm = ChatGoogleGenerativeAI(
    model=MODELO_LLM,
    temperature=0.2,
    google_api_key=api_key,
    max_retries=1,
    timeout=40,
)

tools = [evaluar_riesgo_neuronal_tool, calcular_capacidad_pago_tool]
checkpointer = MemorySaver()

system_prompt = (
    "Eres un Asesor Financiero Inteligente de Microcréditos para PyMEs.\n"
    "Tu objetivo es ayudar a analizar solicitudes de crédito de pequeñas empresas de forma clara y rigurosa.\n\n"
    "Instrucciones de actuación:\n"
    "1. Cuando el usuario te entregue datos financieros de una empresa (ingresos, monto, plazo, antigüedad, endeudamiento, score), "
    "debes invocar la herramienta `evaluar_riesgo_neuronal_tool` para consultar la predicción de la Red Neuronal.\n"
    "2. Si faltan datos para usar una herramienta, pídeselos al usuario; nunca los inventes.\n"
    "3. Cuando requieras validar si la PyME puede asumir la cuota o simular la amortización, invoca la herramienta `calcular_capacidad_pago_tool`.\n"
    "4. Responde siempre con un tono profesional, empático y estructurado, explicando el porqué de la decisión y sus métricas."
)

try:
    agente_pyme = create_react_agent(
        model=llm, tools=tools, checkpointer=checkpointer, prompt=system_prompt
    )
except TypeError:
    agente_pyme = create_react_agent(
        model=llm, tools=tools, checkpointer=checkpointer, state_modifier=system_prompt
    )


def extraer_texto(mensaje) -> str:
    """Los modelos Gemini 3 pueden devolver `content` como lista de bloques; esto lo normaliza a texto."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    partes = []
    for bloque in contenido:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif isinstance(bloque, dict) and bloque.get("type") == "text":
            partes.append(bloque.get("text", ""))
    return "".join(partes)


# --- Modelo de respaldo para cuando el principal está saturado (error 503) ---
MODELO_RESPALDO = os.getenv("GEMINI_MODEL_RESPALDO", "gemini-3.6-flash")
llm_respaldo = ChatGoogleGenerativeAI(
    model=MODELO_RESPALDO, temperature=0.2, google_api_key=api_key,
    max_retries=1, timeout=40,
)
# Comparte el mismo checkpointer: así continúa la MISMA conversación.
try:
    agente_respaldo = create_react_agent(
        model=llm_respaldo, tools=tools, checkpointer=checkpointer, prompt=system_prompt
    )
except TypeError:
    agente_respaldo = create_react_agent(
        model=llm_respaldo, tools=tools, checkpointer=checkpointer, state_modifier=system_prompt
    )


def _es_error_temporal(e: Exception) -> bool:
    texto = str(e)
    claves = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded",
              "timeout", "Timeout", "timed out", "DEADLINE_EXCEEDED")
    return any(c in texto for c in claves)


def ejecutar_agente(mensaje_usuario: str, thread_id: str = "sesion_demo", reintentos: int = 2):
    import time

    config = {"configurable": {"thread_id": thread_id}}
    entrada = {"messages": [("user", mensaje_usuario)]}

    # 1) Modelo principal: pocos reintentos y espera corta (2 s)
    for intento in range(reintentos):
        inicio = time.time()
        try:
            r = agente_pyme.invoke(entrada, config=config)
            print(f"[agente] {MODELO_LLM} respondió en {time.time() - inicio:.1f} s")
            return r
        except Exception as e:
            print(f"[agente] {MODELO_LLM} falló tras {time.time() - inicio:.1f} s "
                  f"(intento {intento + 1}/{reintentos}): {str(e)[:120]}")
            if not _es_error_temporal(e):
                raise
            # El mensaje del usuario ya quedó guardado en la memoria del hilo.
            # Con entrada=None LangGraph reanuda desde ahí sin duplicarlo.
            entrada = None
            time.sleep(2)

    # 2) Si sigue saturado, se pasa al modelo de respaldo y se reanuda el mismo hilo
    print(f"[agente] Pasando al modelo de respaldo {MODELO_RESPALDO}")
    inicio = time.time()
    r = agente_respaldo.invoke(entrada, config=config)
    print(f"[agente] {MODELO_RESPALDO} respondió en {time.time() - inicio:.1f} s")
    return r