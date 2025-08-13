import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Limpeza de linhas com campos nulos
df = pd.read_csv('telecomx_normalizado.csv')
df_clean = df.dropna()
log_output = []
log_output.append(f"Linhas removidas: {len(df) - len(df_clean)}\n")

# Verifique as colunas disponíveis
log_output.append("Colunas disponíveis: " + str(df_clean.columns.tolist()) + "\n")

# Ajuste o nome da coluna de churn conforme necessário
churn_col = None
for col in df_clean.columns:
    if col.lower() == 'churn':
        churn_col = col
        break

if churn_col is None:
    raise ValueError("Coluna 'churn' não encontrada no arquivo. Verifique o nome da coluna.")

# Remove customerID se existir
if 'customerID' in df_clean.columns:
    X = df_clean.drop([churn_col, 'customerID'], axis=1)
    customer_ids = df_clean['customerID']
else:
    X = df_clean.drop(churn_col, axis=1)
    customer_ids = pd.Series(np.arange(len(df_clean)), name='customerID')
y = df_clean[churn_col]

# Se houver colunas não numéricas, faça o encoding
X = pd.get_dummies(X, drop_first=True)

# Para regressão, converta y para numérico
y_reg = y.map({'No': 0, 'Yes': 1})

# Separação treino/teste
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42
)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Modelos
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinamento
logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 3) Modelo preditivo de churn
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Identificando os 100 clientes com maior probabilidade de churn
report_df = X_test.copy()
report_df['customerID'] = ids_test.values
report_df['churn_proba'] = y_proba
top_100_churn = report_df.sort_values('churn_proba', ascending=False).head(100)

# 4) Relatório detalhado
log_output.append("\n--- Relatório Quantitativo ---\n")
log_output.append("Random Forest Classification Report:\n")
log_output.append(classification_report(y_test, y_pred) + "\n")
log_output.append("Confusion Matrix:\n")
log_output.append(str(confusion_matrix(y_test, y_pred)) + "\n")
log_output.append(f"ROC AUC: {roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), y_proba):.2f}\n")

# Regressão Linear para análise de relação entre variáveis e churn
linreg = LinearRegression()
linreg.fit(X_train, y_train_reg)
y_pred_lr = linreg.predict(X_test)
log_output.append(f"Linear Regression MSE: {mean_squared_error(y_test_reg, y_pred_lr):.4f}\n")

# 5) Relatório Qualitativo e Soluções
log_output.append("\n--- Relatório Qualitativo ---\n")
log_output.append("Top 100 clientes com maior risco de churn:\n")
log_output.append(top_100_churn[['customerID', 'churn_proba']].to_string(index=False) + "\n")

# Salva o relatório detalhado dos clientes em CSV
top_100_churn.to_csv('top_100_churn_clients.csv', index=False)

log_output.append("\nDescrição detalhada da predição:\n")
log_output.append("O modelo Random Forest identificou os 100 clientes acima como os mais propensos a churn, com base em padrões históricos.\n")

log_output.append("\nSoluções para reverter o churn:\n")
log_output.append("- Oferecer planos personalizados para clientes de alto risco.\n")
log_output.append("- Melhorar o atendimento ao cliente para segmentos críticos.\n")
log_output.append("- Monitorar métricas de uso e satisfação em tempo real.\n")
log_output.append("- Implementar campanhas de retenção direcionadas.\n")

# Visualização das features mais importantes
importances = rf.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title('Importância das Features para Churn')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

log_output.append("\nGráfico de importância das features salvo como 'feature_importance.png'.\n")

# Salva o relatório no README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.writelines(log_output)