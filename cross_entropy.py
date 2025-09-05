# Implementación de la función de pérdida Cross-Entropy
# 

import numpy as np
def cross_entropy(y_true, y_pred):
    # Asegura que no haya log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred))
