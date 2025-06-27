# Evaluación del modelo con datos externos

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, fbeta_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# === Clase personalizada para umbral personalizado ===
class ThresholdSVMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def get_threshold(self):
        return self.threshold

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params.pop('threshold')
        self.model.set_params(**params)
        return self

    def get_params(self, deep=True):
        return {'model': self.model, 'threshold': self.threshold}

# === Cargar modelo con umbral personalizado ===
model = joblib.load("svm_rbf_model_umbral_0177.pkl")

# === Cargar nuevo dataset etiquetado ===
df = pd.read_parquet("driams2018.parquet")
X_new = df.iloc[:, 1:6001]
y_new = df.iloc[:, 6001]

# === Predicción ===
y_proba = model.predict_proba(X_new)[:, 1]
y_pred = model.predict(X_new)

# === Evaluación y exportación de métricas ===
cm = confusion_matrix(y_new, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) else np.nan

metrics = {
    "Accuracy": accuracy_score(y_new, y_pred),
    "Precision": precision_score(y_new, y_pred),
    "Recall (Sensibilidad)": recall_score(y_new, y_pred),
    "Specificity": specificity,
    "F1 Score": f1_score(y_new, y_pred),
    "F2 Score": fbeta_score(y_new, y_pred, beta=2),
    "AUC": roc_auc_score(y_new, y_proba),
    "MCC": matthews_corrcoef(y_new, y_pred)
}

# Crear DataFrame de métricas
metrics_df = pd.DataFrame([metrics])
metrics_df.to_excel("svm_metrics_umbral0177.xlsx", index=False)

# === Reporte de clasificación (en consola y a Excel) ===
report = classification_report(y_new, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_excel("svm_classification_report_umbral0177.xlsx")

# === Guardar predicciones ===
df_preds = pd.DataFrame({
    'Probabilidad_Resistente': y_proba,
    'Predicción_Binaria': y_pred,
    'Etiqueta_Real': y_new.values
})
df_preds.to_excel("predicciones_svm_umbral0177.xlsx", index=False)

# === Graficar matriz de confusión ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Sensible", "Resistente"],
            yticklabels=["Sensible", "Resistente"])
plt.title("Matriz de Confusión - SVM (umbral 0.177)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("confusion_matrix_svm_0177.png")
plt.show()
