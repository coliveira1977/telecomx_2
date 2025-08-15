import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Carrega e limpa os dados
df = pd.read_csv('telecomx_normalizado.csv')
df_clean = df.dropna()

# Identifica a coluna de churn
churn_col = None
for col in df_clean.columns:
    if col.lower() == 'churn':
        churn_col = col
        break
if churn_col is None:
    raise ValueError("Coluna 'churn' não encontrada no arquivo.")

# Remove customerID se existir
if 'customerID' in df_clean.columns:
    X = df_clean.drop([churn_col, 'customerID'], axis=1)
else:
    X = df_clean.drop(churn_col, axis=1)
y = df_clean[churn_col]

# Encoding de variáveis categóricas
X = pd.get_dummies(X, drop_first=True)
y_bin = y.map({'No': 0, 'Yes': 1})

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Treinamento da Regressão Logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predições
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# Métricas
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Coeficientes
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': logreg.coef_[0]
}).sort_values(by='Coeficiente', key=abs, ascending=False)

# Salva gráfico dos coeficientes
plt.figure(figsize=(10, 8))
sns.barplot(x='Coeficiente', y='Feature', data=coef_df.head(15))
plt.title('Top 15 Coeficientes da Regressão Logística')
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png')
plt.close()

# Gera relatório em markdown
regressao_md = []
regressao_md.append("\n---\n")
regressao_md.append("## 6. Análise de Regressão Logística\n\n")
regressao_md.append("A regressão logística foi utilizada para identificar o impacto das variáveis no risco de churn.\n\n")
regressao_md.append(f"**Acurácia:** {acc:.2f}\n\n")
regressao_md.append(f"**ROC AUC:** {roc_auc:.2f}\n\n")
regressao_md.append("**Classification Report:**\n")
regressao_md.append("```\n" + report + "```\n\n")
regressao_md.append("**Confusion Matrix:**\n")
regressao_md.append("```\n" + str(cm) + "\n```\n\n")
regressao_md.append("**Top 15 Coeficientes (em valor absoluto):**\n\n")
regressao_md.append(coef_df.head(15).to_markdown(index=False))
regressao_md.append("\n\nO gráfico dos coeficientes foi salvo como `logistic_regression_coefficients.png`.\n")

# Adiciona o resultado ao final do README.md
with open("README.md", "a", encoding="utf-8") as f:
    f.writelines(regressao_md)