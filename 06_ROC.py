import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Establecer la semilla aleatoria
np.random.seed(0)

# Generar etiquetas de clase
y = np.random.randint(0, 2, 1000)

# Generar puntacion de prediccion
y_scores_random = np.random.rand(len(y))

# Ordenar puntuaciones y las verdaderas etiquetas en orden descendente de puntacion
sort_indices = np.argsort(y_scores_random)[::-1]
y_sorted = y[sort_indices]

# Calcular Verdaderos positivos y falsos positivos acumulados
TP_cumsum = np.cumsum(y_sorted)
FP_cumsum = np.cumsum(1-y_sorted)

# Calcular TPR y FPR
TPR = TP_cumsum / TP_cumsum[-1]
FPR = FP_cumsum / FP_cumsum[-1]

# Calcular AUC 
AUC = integrate.trapezoid(TPR, FPR)

# Plot ROC curve 
plt.plot(FPR, TPR)
plt.plot([0,1], [0,1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve, AUC = {:.2f}".format(AUC))
plt.show()