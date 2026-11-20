import json
from langchain_core.tools import tool
from src.inferencia import evaluar_riesgo_pyme

@tool
def evaluar_riesgo_neuronal_tool(
    ingresos_mensuales: float,
    monto_solicitado: float,
    plazo_meses: float,
    antiguedad_anos: float,
    ratio_endeudamiento: float,
    score_credito: float
) -> str:
    """Evalúa el nivel de riesgo crediticio de una PyME utilizando la Red Neuronal entrenada.

    Args:
        ingresos_mensuales: Ingresos mensuales promedios de la PyME.
        monto_solicitado: Monto del crédito solicitado.
        plazo_meses: Plazo de pago en meses.
        antiguedad_anos: Años de operación de la empresa.
        ratio_endeudamiento: Proporción de ingresos comprometidos de 0.0 a 1.0.
        score_credito: Puntaje crediticio de 300 a 850.
    """
    res = evaluar_riesgo_pyme(
        ingresos_mensuales=float(ingresos_mensuales),
        monto_solicitado=float(monto_solicitado),
        plazo_meses=int(plazo_meses),
        antiguedad_anos=float(antiguedad_anos),
        ratio_endeudamiento=float(ratio_endeudamiento),
        score_credito=int(score_credito)
    )
    return json.dumps(res, ensure_ascii=False)

@tool
def calcular_capacidad_pago_tool(
    ingresos_mensuales: float,
    ratio_endeudamiento: float,
    monto_solicitado: float,
    plazo_meses: float
) -> str:
    """Simula la capacidad de pago y calcula la cuota mensual para verificar si la PyME puede asumir el crédito.

    Args:
        ingresos_mensuales: Ingresos mensuales promedios de la PyME.
        ratio_endeudamiento: Endeudamiento actual de 0.0 a 1.0.
        monto_solicitado: Monto total del crédito.
        plazo_meses: Plazo en meses.
    """
    ingresos = float(ingresos_mensuales)
    ratio = float(ratio_endeudamiento)
    monto = float(monto_solicitado)
    plazo = int(plazo_meses)
    
    tasa_mensual = 0.018  # 1.8% mensual
    i = tasa_mensual
    n = plazo
    
    cuota = monto * (i * (1 + i)**n) / ((1 + i)**n - 1)
    ingresos_libres = ingresos * (1 - ratio)
    capacidad_maxima_cuota = ingresos_libres * 0.40
    suficiente = cuota <= capacidad_maxima_cuota
    
    res = {
        "cuota_mensual_estimada": round(cuota, 2),
        "ingresos_libres_disponibles": round(ingresos_libres, 2),
        "cuota_maxima_soportable": round(capacidad_maxima_cuota, 2),
        "capacidad_pago_suficiente": suficiente,
        "observacion": "La cuota estimada está dentro del margen de liquidez." if suficiente else "La cuota supera el 40% de los ingresos disponibles."
    }
    return json.dumps(res, ensure_ascii=False)