Objetivos do Desafio
Preparar os dados para a modelagem (tratamento, encoding, normalização).

Realizar análise de correlação e seleção de variáveis.

Treinar dois ou mais modelos de classificação.

Avaliar o desempenho dos modelos com métricas.

Interpretar os resultados, incluindo a importância das variáveis.

Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.


Linhas removidas: 224
Colunas disponíveis: ['customerID', 'Churn', 'customer_gender', 'customer_SeniorCitizen', 'customer_Partner', 'customer_Dependents', 'customer_tenure', 'phone_PhoneService', 'phone_MultipleLines', 'internet_InternetService', 'internet_OnlineSecurity', 'internet_OnlineBackup', 'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV', 'internet_StreamingMovies', 'account_Contract', 'account_PaperlessBilling', 'account_PaymentMethod', 'account_Charges_Monthly', 'account_Charges_Total']

--- Relatório Quantitativo ---
Random Forest Classification Report:
              precision    recall  f1-score   support

          No       0.82      0.93      0.87      1036
         Yes       0.68      0.43      0.53       373

    accuracy                           0.80      1409
   macro avg       0.75      0.68      0.70      1409
weighted avg       0.78      0.80      0.78      1409

Confusion Matrix:
[[960  76]
 [212 161]]
ROC AUC: 0.83
Linear Regression MSE: 0.3095


- **ROC AUC:** 0.83  
- **Linear Regression MSE:** 0.3095

---

## 3. Relatório Qualitativo

### Top 100 Clientes com Maior Risco de Churn

| customerID  | churn_proba |
|-------------|-------------|
| 2737-YNGYW  | 0.950000    |
| 3878-AVSOQ  | 0.923333    |
| 8000-REIQB  | 0.921333    |
| 9488-HGMJH  | 0.916667    |
| 4614-NUVZD  | 0.913333    |
| 4988-IQIGL  | 0.906000    |
| 9728-FTTVZ  | 0.900833    |
| 9985-MWVIX  | 0.900000    |
| 6567-HOOPW  | 0.900000    |
| 6856-RAURS  | 0.885000    |
| ...         | ...         |
| 3990-QYKBE  | 0.700000    |
| 4706-AXVKM  | 0.700000    |

> **Obs:** A lista completa está disponível no arquivo `top_100_churn_clients.csv`.

---

### Descrição Detalhada da Predição

O modelo **Random Forest** identificou os 100 clientes acima como os mais propensos a churn, com base em padrões históricos.

---

### Soluções para Reverter o Churn

- Oferecer planos personalizados para clientes de alto risco.
- Melhorar o atendimento ao cliente para segmentos críticos.
- Monitorar métricas de uso e satisfação em tempo real.
- Implementar campanhas de retenção direcionadas.

---

## 4. Importância das Features

O gráfico de importância das features foi salvo como:  
**`feature_importance.png

Descrição detalhada da predição:
O modelo Random Forest identificou os 100 clientes acima como os mais propensos a churn, com base em padrões históricos.

Soluções para reverter o churn:
- Oferecer planos personalizados para clientes de alto risco.
- Melhorar o atendimento ao cliente para segmentos críticos.
- Monitorar métricas de uso e satisfação em tempo real.
- Implementar campanhas de retenção direcionadas.

Gráfico de importância das features salvo como 'feature_importance.png'.
