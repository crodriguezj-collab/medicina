# Optimización del modelo SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, make_scorer
)
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score
#from sklearn.utils.fixes import loguniform
from imblearn.over_sampling import SMOTE
import joblib
import os

# === Cargar datos ===
df = pd.read_parquet('driams2015_2017.parquet')
X = df.iloc[:, 1:6001]
y = df.iloc[:, 6001]

# === División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Balanceo con SMOTE ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === Búsqueda de hiperparámetros ===
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}

model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
grid = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train_bal, y_train_bal)
best_model = grid.best_estimator_
joblib.dump(best_model, 'svm_rbf_model_optimizado.pkl')

# === Predicción y evaluación ===
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# === Calcular F2 Score ===
def f2_score(y_true, y_pred):
    return f1_score(y_true, y_pred, beta=2)

f2 = fbeta_score(y_test, y_pred, beta=2)
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Guardar métricas ===
df_metrics = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'F2-score', 'AUC'],
    'Valor': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        f2,
        roc_auc
    ]
}).round(3)

# === Guardar curva ROC ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC - SVM RBF Optimizado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_svm_rbf.png")
plt.close()

# === Matriz de confusión ===
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Sensibles', 'Resistentes'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - SVM RBF Optimizado")
plt.tight_layout()
plt.savefig("cm_svm_rbf.png")
plt.close()

# === Guardar en Excel ===
with pd.ExcelWriter("metricas_svm_rbf_optimizado.xlsx", engine='xlsxwriter') as writer:
    df_metrics.to_excel(writer, sheet_name="Métricas", index=False)
    worksheet = writer.sheets["Métricas"]
    worksheet.insert_image("E2", "roc_svm_rbf.png")
    worksheet.insert_image("E22", "cm_svm_rbf.png")

print("Modelo y métricas guardadas correctamente.")
