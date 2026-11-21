Asesor Financiero Inteligente para PyMEs

Sección 1 - Identificación, Ruta y Propuesta de Valor.



Integrantes.



Johan Sebastian Morantes Infanta (55222515)

Francy Daniela Olmos Vaca (55222017)

Darwin Andres Saavedra Araque (55221510)

Sección 2 - Arquitectura del Sistema (Diagrama Propio)

Usuario

app.py (Streamlit)

src/agente.py (AgenteReAct LangGraph + Gemini)

MemorySaver (por thread_id)

evaluar_riesgo_neuronal_tool

calcular_capacidad_pago_tool

src/inferencia.py

models/modelo_microcredito.keras

models/scaler_microcredito.pkl



Estructura del repositorio.



Asesor-Financiero-Inteligen te-para-PyMEs/

├── app.py # Interfaz de chat (Streamlit)

├── requirements.txt

├── .env.example

├── .gitignore

├── assets/

│ └── curvas_entrenamiento_u4.png

├── models/

│ ├── modelo_microcredito.keras # Modelo 2 (Mejorado)

│ └── scaler_microcredito.pkl # StandardScaler

├── notebooks/

│ ├── entrenamiento_u4.ipynb

│ └── evaluacion_agente_u5.ipynb

└── src/

├── agente.py # Agente, reintentos y modelo de respaldo

├── inferencia.py # Carga del modelo y predicción

└── tools.py # Herramientas @tool

Sección 3 - Ingeniería del Modelo de Red Neuronal (Unidad 4) y Análisis de un Fallo Real.



Fuente: notebooks/entrenamiento_u4.ipynb



Variables de entrada (docstring de evaluar_riesgo_neuronal_tool en src/tools.py).



Variable Descripción

ingresos_mensuales Ingresos mensuales promedio de la PyME.

monto_solicitado Monto del crédito solicitado.

plazo_meses Plazo de pago en meses.

antiguedad_anos Años de operación de la empresa.

ratio_endeudamiento Ratio de endeudamiento (proporción de ingresos comprometidos) [0.0 - 1.0].

score_credito Puntaje crediticio [300 - 850].



Datos.



Elemento Valor en el notebook

Origen Dataset sintético (np.random.seed(42))

Muestras 2000

ingresos_mensuales uniforme [3000, 50000]

monto_solicitado uniforme [1000, 30000]

plazo_meses En {6, 12, 18, 24, 36, 48}

antiguedad_anos uniforme [0.5, 15]

ratio_endeudamiento uniforme [0.1, 0.85]

score_credito int [300, 849]

Proporción de clases (riesgo_alto) 1: 0.633, 0: 0.367

Partición train_test_split (estratificado) usando test_size=0.3 y luego test_size=0.5 (prueba tiene 300 registros).



Hiperparámetros y arquitecturas.



Modelo 1 (Base) Modelo 2 (Mejorado)

Capas Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid) Dense(32, relu) → BatchNormalization → Dropout(0.3) → Dense(16, relu) → Dropout(0.2) → Dense(1, sigmoid)

Optimizador adam adam

Pérdida binary_crossentropy binary_crossentropy

Métrica accuracy accuracy

Épocas / batch 40 / 32 40 / 32



Modelo 2 almacenado en models/modelo_microcredito.keras.



Curvas de entrenamiento vs validación.



Curvas de entrenamiento.



Salida del notebook (evidencia de fallo real).



Total de fallos en el test: 33 de 300.

Índice (Name): 1601

valor_real 1

prediccion 0

probabilidad_riesgo 0.313377

ingresos_mensuales 18198.463419

monto_solicitado 17555.445088

plazo_meses 48

antiguedad_anos 2.804409

ratio_endeudamiento 0.180476

score_credito 825

Sección 4 - Diseño del Agente y Matriz de Evaluación (Unidad 5).



Herramientas (src/tools.py)



Herramienta Qué ejecuta Constantes en el código

evaluar_riesgo_neuronal_tool Ejecuta src/inferencia.py (escala 6 variables con el scaler almacenado y evalúa el modelo) Threshold de decisión: Probabilidad > 0.5 → Riesgo Alto (Rechazado), sino Riesgo Bajo (Aprobado).

calcular_capacidad_pago_tool Calcula la cuota (con fórmula de cuota fija), los ingresos libres y la capacidad maxima de cuota. Tasa mensual 0.018. Capacidad maxima de cuota = Ingresos libres 0.40.



Memoria y sesión.



Elemento Dónde

Checkpointer MemorySaver (en src/agente.py)

thread_id Generado por sesión (en app.py). Al pulsar "Limpiar Conversación" se genera un nuevo thread_id (uuid).



Manejo de errores del LLM (en ejecutar_agente, src/agente.py): se intentará 2 veces con el modelo principal. En caso de error temporal (503,429,RESOURCE_EXHAUSTED,timeout) se retoma el mismo hilo con el modelo de respaldo.



Sección 5 - Bitácora de Vibe Coding y Razonamiento Crítico.

Sección 6 - Demostración de la Interfaz (Streamlit / Gradio).

Sección 7 - Guía de Instalación y Ejecución Reproducible.

bash

1. Clonar el repositorio

git clone https://github.com/FrancyDaniela/Asesor-Financiero-Inteligente-para-PyMEs.git

cd Asesor-Financiero-Inteligente-para-PyMEs



2. Crear y activar un entorno virtual

python -m venv venv

source venv/bin/activate # Linux / macOS

venv\Scripts\activate # Windows



3. Instalar dependencias

pip install -r requirements.txt



4. Crear el archivo .env a partir de la plantilla

cp .env.example .env # Windows: copy .env.example .env

# y escribir la clave real en GOOGLE_API_KEY



5. Ejecutar la aplicación

streamlit run app.py

Variable Obligatoria Uso

GOOGLE_API_KEY Sí Clave de Google AI Studio

GEMINI_MODEL No Modelo principal de Gemini

GEMINI_MODEL_RESPALDO No Modelo de respaldo



Dependencias (requirements.txt): tensorflow, scikit-learn, pandas, numpy, matplotlib, seaborn, langchain, langgraph, langchain-google-genai, streamlit, python-dotenv, jupyter, joblib.



Para reentrenar el modelo hay que abrir el notebook notebooks/entrenamiento_u4.ipynb (desde la carpeta notebooks/) y ejecutar las celdas en orden. El notebook almacena el modelo y la gráfica (con rutas relativas ../models y ../assets).
