# Benchmark Results – Driver Risk Prediction

**Task:** Multi-class classification of accident risk level (High / Medium / Low)  
**Dataset:** ~12,300 samples, 30+ engineered features  
**Classes (test set):** High: 62 | Medium: 1,892 | Low: 1,741 (heavily imbalanced)  
**Evaluation:** Hold-out test set (30%), stratified split  
**Metrics:** Accuracy, Macro F1, ROC-AUC (One-vs-Rest)

---

## Leaderboard (Test Set – Final Results)

| Rank | Model                      | Accuracy | Macro F1 | ROC-AUC (OvR) | Notes                        |
|------|----------------------------|----------|----------|---------------|------------------------------|
| 1    | **Logistic Regression**    | **0.9773** | **0.8641** | **0.9932**    | Surprisingly strongest baseline |
| 2    | XGBoost                    | 0.9667   | 0.8161   | 0.9898        | Very close second            |
| 3    | Random Forest              | 0.9172   | 0.7631   | 0.9804        | Good but clearly beaten      |

---

## Detailed Classification Reports

### 1. Logistic Regression (Best Baseline)

```
              precision    recall  f1-score   support

        High       0.53      0.76      0.62        62
         Low       0.99      0.99      0.99      1741
      Medium       0.99      0.97      0.98      1892

    accuracy                           0.98      3695
```

---

### 2. XGBoost

```
              precision    recall  f1-score   support

        High       0.56      0.45      0.50        62
         Low       0.98      0.99      0.98      1741
      Medium       0.97      0.97      0.97      1892

    accuracy                           0.97      3695
```

---

### 3. Random Forest

```
              precision    recall  f1-score   support

        High       0.51      0.39      0.44        62
         Low       0.90      0.96      0.93      1741
      Medium       0.94      0.89      0.92      1892

    accuracy                           0.92      3695
```

---

## Annotated Test Results 

*Note: These results are from an initial smaller validation set with limited feature engineering.*

### Random Forest
**Accuracy:** 0.6000

```
              precision    recall  f1-score   support

        HIGH       0.57      0.73      0.64        11
         LOW       1.00      0.43      0.60         7
      MEDIUM       0.54      0.58      0.56        12

    accuracy                           0.60        30
   macro avg       0.70      0.58      0.60        30
weighted avg       0.66      0.60      0.60        30
```

---

### Logistic Regression
**Accuracy:** 0.3667

```
              precision    recall  f1-score   support

        HIGH       0.20      0.18      0.19        11
         LOW       0.60      0.43      0.50         7
      MEDIUM       0.40      0.50      0.44        12

    accuracy                           0.37        30
   macro avg       0.40      0.37      0.38        30
weighted avg       0.37      0.37      0.36        30
```

---

### XGBoost
**Accuracy:** 0.8333

```
              precision    recall  f1-score   support

        HIGH       0.91      0.91      0.91        11
         LOW       0.83      0.71      0.77         7
      MEDIUM       0.77      0.83      0.80        12

    accuracy                           0.83        30
   macro avg       0.84      0.82      0.83        30
weighted avg       0.84      0.83      0.83        30
```

---

## Final Benchmark Summary (Annotated Set)

| Model               | Accuracy | F1 (macro) | ROC-AUC |
|---------------------|----------|------------|---------|
| Random Forest       | 0.6000   | 0.6000     | 0.7114  |
| Logistic Regression | 0.3667   | 0.3783     | 0.5241  |
| XGBoost             | 0.8333   | 0.8261     | 0.9234  |

---

**Key Observations:**
- The main test set (n=3,695) shows strong performance across all models with extensive feature engineering
- The annotated small holdout set (n=30) reveals more realistic performance on limited features
- XGBoost demonstrates consistent strength across both evaluation scenarios
- Class imbalance remains a challenge, particularly for the minority "High" risk class
