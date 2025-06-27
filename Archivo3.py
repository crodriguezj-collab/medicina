# Modelo inicial de regresión logística
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

# Cargar el archivo de Excel
file_path = 'driams2015_2017.xlsx' 
df = pd.read_excel(file_path)

# Extraer features y label
X = df.iloc[:, 1:6001]    # bin_0 to bin_5999
y = df.iloc[:, 6001]      # oxacillin susceptibility (0 or 1)

# División Train-test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo de regresión logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predecir probabilidades y labels de clase
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# ROC curve y AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

# Otras métricas de desempeño
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Imprimir todas las métricas
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f} (Sensitivity)")
print(f"Specificity: {specificity:.3f}")
print(f"F1-score:    {f1:.3f}")
print(f"AUC:         {auc_score:.3f}")

# Plot curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Oxacillin Susceptibility Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
