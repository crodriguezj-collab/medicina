# Evaluación del modelo optimizado con el umbral de 0.177 sobre los datos originales completos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score
)
import joblib

# === Cargar modelo y datos ===
model = joblib.load("svm_rbf_model_optimizado.pkl")
df = pd.read_parquet("driams2015_2017.parquet")

X = df.iloc[:, 1:6001]
y = df.iloc[:, 6001]

# === Predicción y clasificación ===
threshold = 0.177
y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# === Cálculo de métricas ===
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
f2 = 5 * (precision * recall) / ((4 * precision) + recall)
accuracy = (y == y_pred).mean()
auc = roc_auc_score(y, y_proba)
cm = confusion_matrix(y, y_pred)

# === Clasificación de predicciones ===
def clasificar_error(real, pred):
    if real == 1 and pred == 1:
        return 'TP'
    elif real == 0 and pred == 0:
        return 'TN'
    elif real == 0 and pred == 1:
        return 'FP'
    elif real == 1 and pred == 0:
        return 'FN'

df_preds = pd.DataFrame({
    'Real': y.values,
    'Probabilidad': y_proba,
    'Predicho': y_pred
})
df_preds['Tipo'] = df_preds.apply(lambda row: clasificar_error(row['Real'], row['Predicho']), axis=1)

# === Gráfico ROC ===
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("1 - Especificidad (FPR)")
plt.ylabel("Sensibilidad (TPR)")
plt.title("Curva ROC - SVM (umbral 0.177)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_svm_0177.png")
plt.close()

# === Histograma de probabilidades por tipo ===
plt.figure(figsize=(8, 5))
for tipo in ['TP', 'FP', 'TN', 'FN']:
    subset = df_preds[df_preds['Tipo'] == tipo]
    plt.hist(subset['Probabilidad'], bins=25, alpha=0.6, label=tipo, edgecolor='black')
plt.xlabel("Probabilidad de resistencia")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades por tipo de predicción")
plt.legend()
plt.tight_layout()
plt.savefig("hist_probabilidades_svm_0177.png")
plt.close()

# === Conteo por tipo de predicción ===
plt.figure(figsize=(6, 4))
df_preds['Tipo'].value_counts().reindex(['TP', 'FP', 'TN', 'FN']).plot(
    kind='bar', color=['green', 'orange', 'blue', 'red'], edgecolor='black'
)
plt.ylabel("Cantidad")
plt.title("Conteo por tipo de predicción")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("conteo_predicciones_svm_0177.png")
plt.close()

# === Exportar métricas y resultados ===
metrics = {
    'Accuracy': round(accuracy, 3),
    'Precision': round(precision, 3),
    'Recall': round(recall, 3),
    'F1 Score': round(f1, 3),
    'F2 Score': round(f2, 3),
    'AUC': round(auc, 3),
    'Umbral': threshold
}
df_metrics = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])
df_metrics.to_excel("metricas_finales_svm_0177.xlsx", index=False)
df_preds.to_excel("predicciones_svm_0177.xlsx", index=False)
pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]).to_excel("matriz_confusion_svm_0177.xlsx")
