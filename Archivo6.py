# Análisis SHAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, fbeta_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# === Cargar datos ===
df = pd.read_parquet('driams2015_2017.parquet')
X = df.iloc[:, 1:6001]
y = df['oxa']

# === División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Pipeline ===
pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=300)),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
])

# === Búsqueda de hiperparámetros ===
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.001],
    'selector__k': [300]
}
grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# === Evaluación ===
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

print("\n--- Evaluación SVM RBF ---")
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"F2 Score: {fbeta_score(y_test, y_pred, beta=2):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"AP (average precision): {average_precision_score(y_test, y_proba):.4f}")

# === Guardar modelo y parámetros ===
joblib.dump(best_model, 'svm_rbf_model.pkl')
with open('svm_rbf_config.json', 'w') as f:
    json.dump({'best_params': grid.best_params_}, f, indent=4)

# === Gráficas ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC - SVM RBF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('svm_rbf_roc_curve.pdf')
plt.close()

prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f'AP = {average_precision_score(y_test, y_proba):.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - SVM RBF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('svm_rbf_prc_curve.pdf')
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Matriz de Confusión - SVM RBF')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.colorbar()
plt.xticks([0, 1], ['Sensible', 'Resistente'])
plt.yticks([0, 1], ['Sensible', 'Resistente'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig('svm_rbf_confusion_matrix.pdf')
plt.close()

# === SHAP Analysis ===

# Extraer pasos del pipeline
selector = best_model.named_steps['selector']
scaler = best_model.named_steps['scaler']
svm_model = best_model.named_steps['svm']

X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)
X_train_sel_scaled = scaler.transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)

# Validaciones
assert X_train_sel_scaled.shape[1] == 300
assert not np.any(np.isnan(X_train_sel_scaled))
assert not np.any(np.isinf(X_train_sel_scaled))

# Background para SHAP
X_background = shap.sample(X_train_sel_scaled, 100, random_state=42)
X_background += 1e-6 * np.random.normal(size=X_background.shape)
X_background = X_background.astype(np.float64)
X_explain = X_test_sel_scaled[:200].astype(np.float64)

# Explainer y valores SHAP
explainer = shap.KernelExplainer(lambda X: svm_model.predict_proba(X)[:, 1], X_background)
shap_values = explainer.shap_values(X_explain, nsamples="auto")

selected_feature_names = [f'bin_{i}' for i in selector.get_support(indices=True)]
if isinstance(shap_values, list):
    shap_array = np.abs(shap_values[1])  # para la clase "resistente"
else:
    shap_array = np.abs(shap_values)

mean_shap = shap_array.mean(axis=0)
shap_importance_df = pd.DataFrame({
    'bin': selected_feature_names,
    'mean_abs_shap': mean_shap
}).sort_values(by='mean_abs_shap', ascending=False)

top20_shap = shap_importance_df.head(20)
top20_shap.to_excel('top20_bins_shap_svm.xlsx', index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top20_shap['bin'][::-1], top20_shap['mean_abs_shap'][::-1])
plt.xlabel("Media del valor absoluto de SHAP")
plt.title("Top 20 bins más influyentes según SHAP")
plt.tight_layout()
plt.savefig("shap_top20_barplot.pdf")
plt.close()

# Guardar resultados
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score', 'MCC', 'AUC', 'AP'],
    'Value': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        fbeta_score(y_test, y_pred, beta=2),
        matthews_corrcoef(y_test, y_pred),
        roc_auc_score(y_test, y_proba),
        average_precision_score(y_test, y_proba)
    ]
})
conf_matrix_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
pred_df = pd.DataFrame({
    'Sample': df.iloc[X_test.index, 0],
    'True Label': y_test.values,
    'Predicted Label': y_pred,
    'Predicted Probability': y_proba
})

with pd.ExcelWriter('svm_model_results.xlsx') as writer:
    metrics_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
    report_df.to_excel(writer, sheet_name='Classification Report')
    conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
    pred_df.to_excel(writer, sheet_name='Predictions', index=False)
    top20_shap.to_excel(writer, sheet_name='Top SHAP Features', index=False)

print("Modelo, métricas, SHAP y predicciones exportadas correctamente.")
