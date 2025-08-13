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

--- Relatório Qualitativo ---
Top 100 clientes com maior risco de churn:
customerID  churn_proba
2737-YNGYW     0.950000
3878-AVSOQ     0.923333
8000-REIQB     0.921333
9488-HGMJH     0.916667
4614-NUVZD     0.913333
4988-IQIGL     0.906000
9728-FTTVZ     0.900833
9985-MWVIX     0.900000
6567-HOOPW     0.900000
6856-RAURS     0.885000
2215-ZAFGX     0.880000
3320-VEOYC     0.870000
5494-HECPR     0.870000
4822-RVYBB     0.870000
9057-SIHCH     0.870000
7409-KIUTL     0.866667
4912-PIGUY     0.860000
8087-LGYHQ     0.860000
2034-GDRCN     0.860000
9605-WGJVW     0.858333
0021-IKXGC     0.856667
1761-AEZZR     0.850000
3707-GNWHM     0.850000
0325-XBFAC     0.840000
8086-OVPWV     0.840000
4510-HIMLV     0.834083
6969-MVBAI     0.830000
2012-NWRPA     0.830000
5567-WSELE     0.830000
1455-UGQVH     0.830000
8884-ADFVN     0.830000
0488-GSLFR     0.820167
1184-PJVDB     0.820000
5797-APWZC     0.820000
3027-ZTDHO     0.820000
0697-ZMSWS     0.820000
5356-RHIPP     0.816061
3370-HXOPH     0.810000
9231-ZJYAM     0.810000
9837-BMCLM     0.810000
4548-SDBKE     0.810000
7526-BEZQB     0.810000
1766-GKNMI     0.810000
6357-JJPQT     0.810000
5348-CAGXB     0.800000
8414-MYSHR     0.800000
8361-LTMKD     0.800000
8875-AKBYH     0.800000
9025-AOMKI     0.800000
0670-KDOMA     0.799583
0023-HGHWL     0.790000
1751-NCDLI     0.790000
1508-DFXCU     0.790000
1567-DSCIC     0.790000
1143-NMNQJ     0.790000
8443-ZRDBZ     0.790000
6680-WKXRZ     0.790000
0637-KVDLV     0.786000
8873-GLDMH     0.785000
7594-RQHXR     0.783333
4581-SSPWD     0.780000
4713-LZDRV     0.780000
3295-YVUSR     0.780000
4927-WWOOZ     0.780000
1866-RZZQS     0.770000
5445-UTODQ     0.770000
5835-BEQEU     0.770000
1569-TTNYJ     0.770000
3254-YRILK     0.770000
2262-SLNVK     0.766667
5032-MIYKT     0.762500
7028-DVOIQ     0.760000
7932-WPTDS     0.760000
0334-GDDSO     0.760000
1086-LXKFY     0.750000
9282-IZGQK     0.750000
7493-TPUWZ     0.750000
6435-SRWBJ     0.750000
6618-RYATB     0.750000
0689-NKYLF     0.750000
5976-JCJRH     0.750000
3841-CONLJ     0.750000
4110-PFEUZ     0.744167
9965-YOKZB     0.740000
3677-IYRBF     0.740000
4597-NUCQV     0.740000
9185-TQCVP     0.730000
7823-JSOAG     0.720000
9685-WKZGT     0.720000
5564-NEMQO     0.720000
2845-AFFTX     0.720000
2800-VEQXM     0.710000
7969-AULMZ     0.710000
3138-BKYAV     0.710000
6615-NGGZJ     0.710000
9944-HKVVB     0.710000
2845-HSJCY     0.710000
0151-ONTOV     0.708167
3990-QYKBE     0.700000
4706-AXVKM     0.700000

Descrição detalhada da predição:
O modelo Random Forest identificou os 100 clientes acima como os mais propensos a churn, com base em padrões históricos.

Soluções para reverter o churn:
- Oferecer planos personalizados para clientes de alto risco.
- Melhorar o atendimento ao cliente para segmentos críticos.
- Monitorar métricas de uso e satisfação em tempo real.
- Implementar campanhas de retenção direcionadas.

Gráfico de importância das features salvo como 'feature_importance.png'.
