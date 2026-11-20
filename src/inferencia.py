import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import joblib 
import os 

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/modelo_microcredito.keras') 
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler_microcredito.pkl') 

_model = None 
_scaler = None 

def _cargar_recursos(): 
    global _model, _scaler 
    if _model is None: 
        _model = keras.models.load_model(MODEL_PATH) 
    if _scaler is None: 
        _scaler = joblib.load(SCALER_PATH) 
        
def evaluar_riesgo_pyme(ingresos_mensuales: float, monto_solicitado: float, plazo_meses: int, antiguedad_anos: float, ratio_endeudamiento: float, score_credito: int) -> dict:
    """ 
    Ejecuta la inferencia de la Red Neuronal para evaluar el riesgo crediticio. 
    """ 
    _cargar_recursos()

    datos_df = pd.DataFrame([{ 
        'ingresos_mensuales': ingresos_mensuales, 
        'monto_solicitado': monto_solicitado,
        'plazo_meses': plazo_meses, 
        'antiguedad_anos': antiguedad_anos, 
        'ratio_endeudamiento': ratio_endeudamiento, 
        'score_credito': score_credito 
    }]) 
     
    datos_scaled = _scaler.transform(datos_df) 
    prob_riesgo_alto = float(_model.predict(datos_scaled, verbose=0)[0][0])
    
    diagnostico = "Riesgo Alto (Rechazado)" if prob_riesgo_alto > 0.5 else "Riesgo Bajo (Aprobado)"
    confianza = prob_riesgo_alto if prob_riesgo_alto > 0.5 else (1 - prob_riesgo_alto)
    
    return { 
        "diagnostico": diagnostico, 
        "probabilidad_riesgo_alto": round(prob_riesgo_alto, 4),
        "nivel_confianza": f"{round(confianza * 100, 2)}%",
        "recomendacion_preliminar": "Aprobación viable sujeta a verificación de capacidad de pago." if prob_riesgo_alto <= 0.5 else "Rechazado por probabilidad de mora elevada." 
        }