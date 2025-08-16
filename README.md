# Relatório de Análise de Churn

## 1. Limpeza de Dados

- **Linhas removidas:** 224
- **Colunas disponíveis:**
  ```
  ['customerID', 'Churn', 'customer_gender', 'customer_SeniorCitizen', 'customer_Partner', 'customer_Dependents', 'customer_tenure', 'phone_PhoneService', 'phone_MultipleLines', 'internet_InternetService', 'internet_OnlineSecurity', 'internet_OnlineBackup', 'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV', 'internet_StreamingMovies', 'account_Contract', 'account_PaperlessBilling', 'account_PaymentMethod', 'account_Charges_Monthly', 'account_Charges_Total']
  ```

---

## 2. Matriz de Correlação

A matriz de correlação das variáveis numéricas foi gerada para identificar relações relevantes. Variáveis com maior correlação (positiva ou negativa) com a evasão (Churn) são candidatas importantes para o modelo preditivo.

O gráfico foi salvo como `correlation_matrix.png`.

---

## 3. Relatório Quantitativo

### Random Forest Classification Report (threshold ajustado)

```
              precision    recall  f1-score   support

          No       0.95      0.29      0.44      1035
         Yes       0.33      0.96      0.49       374

    accuracy                           0.47      1409
   macro avg       0.64      0.62      0.47      1409
weighted avg       0.78      0.47      0.46      1409
```

**Confusion Matrix:**
```
[[300 735]
 [ 16 358]]
```

- **ROC AUC:** 0.82
- **Recall classe YES:** 0.96
- **F1-score classe YES:** 0.49
- **Linear Regression MSE:** 0.2249

---

## 4. Relatório Qualitativo

### Top 100 Clientes com Maior Risco de Churn

customerID  churn_proba
2636-ALXXZ     0.667351
3988-RQIXO     0.664224
8414-MYSHR     0.661543
8245-UMPYT     0.660537
3320-VEOYC     0.660091
1704-NRWYE     0.658948
5935-FCCNB     0.657248
1564-NTYXF     0.656822
0722-TROQR     0.655259
8051-HJRLT     0.654808
2055-PDADH     0.654808
5567-WSELE     0.654230
2506-TNFCO     0.653694
1016-DJTSV     0.652127
9685-WKZGT     0.650675
8149-RSOUN     0.650156
4910-GMJOT     0.650156
1363-TXLSL     0.648181
2672-HUYVI     0.647293
8739-XNIKG     0.645517
4145-UQXUQ     0.645038
7450-NWRTR     0.643992
1273-MTETI     0.643821
7206-PQBBZ     0.642920
8875-AKBYH     0.642801
4529-CKBCL     0.642654
7760-OYPDY     0.642654
6357-JJPQT     0.642654
6474-FVJLC     0.642143
5229-PRWKT     0.642143
2691-NZETQ     0.641880
4210-QFJMF     0.641537
2004-OCQXK     0.641537
4614-NUVZD     0.641537
2609-IAICY     0.641029
8808-ELEHO     0.639819
4415-IJZTP     0.639147
9306-CPCBC     0.639147
3199-XGZCY     0.639147
1320-HTRDR     0.639147
9878-TNQGW     0.639066
2868-MZAGQ     0.639066
6521-YYTYI     0.638759
7074-IEVOJ     0.638364
1640-PLFMP     0.637893
4282-YMKNA     0.637766
3398-FSHON     0.637470
0122-OAHPZ     0.637381
4229-CZMLL     0.636891
5835-BEQEU     0.636891
0637-KVDLV     0.636891
0404-SWRVG     0.636891
8884-ADFVN     0.636567
7245-NIIWQ     0.636285
9408-SSNVZ     0.636285
5228-EXCET     0.636120
0195-IESCP     0.634903
9957-YODKZ     0.634542
7249-WBIYX     0.634300
4795-KTRTH     0.633897
2568-BRGYX     0.633814
9717-QEBGU     0.633814
3871-IKPYH     0.633603
8010-EZLOU     0.633272
8270-RKSAP     0.632432
9681-OXGVC     0.632168
7660-HDPJV     0.632159
9728-FTTVZ     0.632159
9947-OTFQU     0.632096
2265-CYWIV     0.632045
3223-DWFIO     0.631439
7341-LXCAF     0.631399
3317-VLGQT     0.631259
1184-PJVDB     0.630957
4060-LDNLU     0.630952
9506-UXUSK     0.630868
3494-JCHRQ     0.630209
5382-SOYZL     0.629930
5419-JPRRN     0.629912
8290-YWKHZ     0.629398
8835-VSDSE     0.629318
9094-AZPHK     0.628543
9253-QXKBE     0.627759
4283-FUTGF     0.627581
2180-DXNEG     0.627505
9490-DFPMD     0.627317
0754-UKWQP     0.626415
1769-GRUIK     0.626200
5348-CAGXB     0.626037
8861-HGGKB     0.625894
6328-ZPBGN     0.625892
8622-ZLFKO     0.625227
0018-NYROU     0.625193
5027-QPKTE     0.625105
9172-ANCRX     0.625042
5701-YVSVF     0.625028
2528-HFYZX     0.624962
9560-ARGQJ     0.624517
8775-ERLNB     0.624371
2737-YNGYW     0.624308

> **Obs:** A lista completa está disponível no arquivo `top_100_churn_clients.csv`.

---

### Descrição Detalhada da Predição

O modelo **Random Forest** foi ajustado para priorizar o acerto da classe YES (clientes que vão dar churn), utilizando balanceamento de classes e threshold reduzido.

Interpretação:

Classe "No" (majoritária)

Precision 0.95 → quando o modelo prevê "No", quase sempre acerta.

Recall 0.29 → porém, o modelo só acerta 29% de todos os "No" reais; ele está prevendo poucos "No".

F1 0.44 → baixa, porque o recall caiu muito.

Classe "Yes" (minoritária)

Precision 0.33 → quando o modelo prevê "Yes", só acerta 33% das vezes.

Recall 0.96 → captura quase todos os "Yes" reais; ou seja, o modelo está prevendo “Yes” muito facilmente.

F1 0.49 → ainda baixo, porque a precisão é baixa.

2️⃣ Acurácia e médias

Accuracy 0.47 → caiu para 47% porque o modelo erra muitas previsões da classe majoritária "No".

Macro avg 0.64 / 0.62 / 0.47 → mostra que o desempenho médio entre as classes ficou equilibrado em recall, mas fraco no F1.

Weighted avg 0.78 / 0.47 / 0.46 → média ponderada pela quantidade de casos (a maioria é "No"), indicando que o modelo está performando mal no geral.

3️⃣ O que está acontecendo

Quando você ajusta o threshold para ser mais “sensível” a detectar a classe minoritária (“Yes”):

O modelo começa a prever muito mais “Yes”.

Isso aumenta o recall da classe “Yes” (96%) — quase nenhum “Yes” real é perdido.

Mas, ao mesmo tempo, erra muitos “No”, porque agora ele classifica erroneamente vários “No” como “Yes”.

Resultado: a acurácia geral cai (47%) e a classe majoritária sofre.

4️⃣ Resumo intuitivo

Antes: o modelo era conservador, acertava "No", perdia "Yes".

Agora: o modelo é agressivo para detectar "Yes", acertando quase todos, mas sacrificando a classe "No".

Trade-off clássico: aumentar recall de uma classe geralmente reduz precisão e acurácia geral.
---

### Soluções para Reverter o Churn

- Oferecer planos personalizados para clientes de alto risco.
- Melhorar o atendimento ao cliente para segmentos críticos.
- Monitorar métricas de uso e satisfação em tempo real.
- Implementar campanhas de retenção direcionadas.

---

## 5. Importância das Features

O gráfico de importância das features foi salvo como:
**`feature_importance.png`**


---
## 6. Análise de Regressão Logística

A regressão logística foi utilizada para identificar o impacto das variáveis no risco de churn. O modelo foi ajustado para priorizar o acerto da classe YES (clientes que vão dar churn), utilizando balanceamento de classes e threshold reduzido.

**Acurácia:** 0.70

**ROC AUC:** 0.84

**Recall classe YES:** 0.86

**F1-score classe YES:** 0.60

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.92      0.64      0.76      1035
           1       0.46      0.86      0.60       374

    accuracy                           0.70      1409
   macro avg       0.69      0.75      0.68      1409
weighted avg       0.80      0.70      0.71      1409
```

**Confusion Matrix:**
```
[[662 373]
 [ 54 320]]
```

**Top 15 Coeficientes (em valor absoluto):**

| Feature                       |   Coeficiente |
|:------------------------------|--------------:|
| account_Charges_Total_1052.35 |       1.67697 |
| account_Charges_Total_1099.6  |       1.67612 |
| account_Charges_Total_3046.4  |       1.67303 |
| account_Charges_Total_20.1    |       1.66075 |
| account_Charges_Total_5154.6  |       1.61468 |
| account_Charges_Total_4820.15 |       1.59533 |
| account_Charges_Total_1327.15 |       1.59301 |
| account_Charges_Total_1021.8  |       1.59117 |
| account_Charges_Total_6579.05 |       1.58243 |
| account_Charges_Total_4481    |       1.56011 |
| account_Charges_Total_2460.15 |       1.53957 |
| account_Charges_Total_740.3   |       1.53927 |
| account_Charges_Total_3563.8  |       1.53452 |
| account_Charges_Total_3147.5  |       1.51793 |
| account_Charges_Total_20.5    |       1.50585 |

O gráfico dos coeficientes foi salvo como `logistic_regression_coefficients.png`.
