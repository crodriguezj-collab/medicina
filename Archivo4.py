# Código para evaluar regresión logística balanceada, con SMOTE, random forest, SVM, XGbBoost y LightGBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# === Cargar datos ===
df = pd.read_parquet('/Users/CARJ/Documents/ProjectAI3/driams.parquet')
X = df.iloc[:, 1:6001]
y = df.iloc[:, 6001]

# === División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Escalado para SVM ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === SMOTE (opcional para Logistic Regression) ===
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# === Modelos ===
models = {
    'LogReg Balanced': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'LogReg SMOTE': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'SVM RBF': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                             use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced',
                               scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42)
}

results = {}
fpr_dict = {}
tpr_dict = {}
auc_dict = {}

for name, model in models.items():
    if name == 'LogReg SMOTE':
        model.fit(X_train_smote, y_train_smote)
        X_eval = X_test
    elif name == 'SVM RBF':
        model.fit(X_train_scaled, y_train)
        X_eval = X_test_scaled
    else:
        model.fit(X_train, y_train)
        X_eval = X_test

    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = model.predict(X_eval)

    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'Precision Clase 1': report['1']['precision'],
        'Recall Clase 1': report['1']['recall'],
        'F1-score Clase 1': report['1']['f1-score'],
        'Exactitud total': report['accuracy']
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fpr_dict[name] = fpr
    tpr_dict[name] = tpr
    auc_dict[name] = auc

# === Mostrar métricas comparativas ===
df_results = pd.DataFrame(results).T.round(3)
print("\nResumen comparativo:\n")
print(df_results)

# === Graficar curvas ROC ===
plt.figure(figsize=(10, 6))
for name in models:
    plt.plot(fpr_dict[name], tpr_dict[name], label=f"{name} (AUC = {auc_dict[name]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC - Comparación de Modelos')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
