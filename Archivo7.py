# Determinación del mejor umbral para optimizar el F2 score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    fbeta_score, precision_score, recall_score
)

# === Cargar modelo y datos ===
model = joblib.load("svm_rbf_model_optimizado.pkl")  
df = pd.read_parquet("driams2015_2017.parquet")       # Dataset etiquetado
X = df.iloc[:, 1:6001]
y = df.iloc[:, 6001]

# === Predicción de probabilidades ===
y_proba = model.predict_proba(X)[:, 1]

# === Evaluación de múltiples umbrales ===
thresholds = np.linspace(0.01, 0.99, 300)
metrics = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f2 = fbeta_score(y, y_pred, beta=2, zero_division=0)
    acc = np.mean(y_pred == y)
    metrics.append((t, precision, recall, f1, f2, acc))

df_metrics = pd.DataFrame(metrics, columns=['Threshold', 'Precision', 'Recall', 'F1', 'F2', 'Accuracy'])

# === Selección del mejor umbral por F2 ===
best = df_metrics.loc[df_metrics['F2'].idxmax()]
print("\n=== Mejor umbral para F2 ===")
print(best)

# === Guardar métricas en Excel ===
df_metrics.to_excel("evaluacion_umbral_svm.xlsx", index=False)

# === Gráficos ===
plt.figure(figsize=(8, 6))
plt.plot(df_metrics['Threshold'], df_metrics['Recall'], label="Recall", color='blue')
plt.plot(df_metrics['Threshold'], df_metrics['Precision'], label="Precision", color='orange')
plt.plot(df_metrics['Threshold'], df_metrics['F1'], label="F1", color='green')
plt.plot(df_metrics['Threshold'], df_metrics['F2'], label="F2", color='purple')
plt.axvline(best['Threshold'], linestyle='--', color='red', label=f"Mejor umbral: {best['Threshold']:.2f}")
plt.xlabel("Umbral de decisión")
plt.ylabel("Valor de la métrica")
plt.title("Evaluación de métricas vs umbral")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("metricas_vs_umbral.png")
plt.show()
