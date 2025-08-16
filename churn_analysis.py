import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 1) Limpeza de linhas com campos nulos
df = pd.read_csv('telecomx_normalizado.csv')
df_clean = df.dropna()
log_output = []
log_output.append(f"# Relatório de Análise de Churn\n\n")
log_output.append(f"## 1. Limpeza de Dados\n\n")
log_output.append(f"- **Linhas removidas:** {len(df) - len(df_clean)}\n")
log_output.append("- **Colunas disponíveis:**\n")
log_output.append(f"  ```\n  {df_clean.columns.tolist()}\n  ```\n\n")

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
    X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42, stratify=y_reg)

# 2) Matriz de correlação e visualização
corr = df_clean.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação das Variáveis Numéricas")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

log_output.append("---\n\n")
log_output.append("## 2. Matriz de Correlação\n\n")
log_output.append("A matriz de correlação das variáveis numéricas foi gerada para identificar relações relevantes. Variáveis com maior correlação (positiva ou negativa) com a evasão (Churn) são candidatas importantes para o modelo preditivo.\n\n")
log_output.append("O gráfico foi salvo como `correlation_matrix.png`.\n\n")

# 3) Modelos com ajuste para classe desbalanceada
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')

# Treinamento
logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 4) Modelo preditivo de churn com threshold ajustado
y_proba = rf.predict_proba(X_test)[:, 1]
threshold = 0.35  # threshold menor para aumentar recall da classe YES
y_pred = np.where(y_proba > threshold, 'Yes', 'No')

# Identificando os 100 clientes com maior probabilidade de churn
report_df = X_test.copy()
report_df['customerID'] = ids_test.values
report_df['churn_proba'] = y_proba
top_100_churn = report_df.sort_values('churn_proba', ascending=False).head(100)

# 5) Relatório detalhado
log_output.append("---\n\n")
log_output.append("## 3. Relatório Quantitativo\n\n")
log_output.append("### Random Forest Classification Report (threshold ajustado)\n\n")
log_output.append("```\n")
log_output.append(classification_report(y_test, y_pred))
log_output.append("```\n\n")
log_output.append("**Confusion Matrix:**\n")
log_output.append("```\n")
log_output.append(str(confusion_matrix(y_test, y_pred)) + "\n")
log_output.append("```\n\n")
log_output.append(f"- **ROC AUC:** {roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), y_proba):.2f}\n")
log_output.append(f"- **Recall classe YES:** {recall_score(y_test.map({'No': 0, 'Yes': 1}), np.where(y_pred=='Yes',1,0)):.2f}\n")
log_output.append(f"- **F1-score classe YES:** {f1_score(y_test.map({'No': 0, 'Yes': 1}), np.where(y_pred=='Yes',1,0)):.2f}\n")
linreg = LinearRegression()
linreg.fit(X_train, y_train_reg)
y_pred_lr = linreg.predict(X_test)
log_output.append(f"- **Linear Regression MSE:** {mean_squared_error(y_test_reg, y_pred_lr):.4f}\n\n")

# 6) Relatório Qualitativo e Soluções
log_output.append("---\n\n")
log_output.append("## 4. Relatório Qualitativo\n\n")
log_output.append("### Top 100 Clientes com Maior Risco de Churn\n\n")
log_output.append(top_100_churn[['customerID', 'churn_proba']].to_string(index=False) + "\n\n")
log_output.append("> **Obs:** A lista completa está disponível no arquivo `top_100_churn_clients.csv`.\n\n")

top_100_churn.to_csv('top_100_churn_clients.csv', index=False)

log_output.append("---\n\n")
log_output.append("### Descrição Detalhada da Predição\n\n")
log_output.append("O modelo **Random Forest** foi ajustado para priorizar o acerto da classe YES (clientes que vão dar churn), utilizando balanceamento de classes e threshold reduzido.\n\n")

log_output.append("---\n\n")
log_output.append("### Soluções para Reverter o Churn\n\n")
log_output.append("- Oferecer planos personalizados para clientes de alto risco.\n")
log_output.append("- Melhorar o atendimento ao cliente para segmentos críticos.\n")
log_output.append("- Monitorar métricas de uso e satisfação em tempo real.\n")
log_output.append("- Implementar campanhas de retenção direcionadas.\n\n")

# 7) Visualização das features mais importantes
importances = rf.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title('Importância das Features para Churn')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

log_output.append("---\n\n")
log_output.append("## 5. Importância das Features\n\n")
log_output.append("O gráfico de importância das features foi salvo como:\n**`feature_importance.png`**\n\n")

# Salva o relatório no README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.writelines(log_output)

# Geração do PDF com gráficos
with PdfPages('relatorio_churn.pdf') as pdf:
    # Matriz de correlação
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de Correlação das Variáveis Numéricas")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    # Importância das features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp, y=feat_imp.index)
    plt.title('Importância das Features para Churn')
    plt.tight_layout()
    pdf.savefig()
    plt.close()