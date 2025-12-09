# Benchmark Results – Driver Risk Prediction 

**Task:** Multi-class classification of accident risk level (High / Medium / Low)  
**Dataset:** ~12,300 samples, 30+ engineered features  
**Classes (test set):** High: 62 | Medium: 1,892 | Low: 1,741 (heavily imbalanced)  
**Evaluation:** Hold-out test set (30%), stratified split  
**Metrics:** Accuracy, Macro F1, ROC-AUC (One-vs-Rest)

## Leaderboard (Test Set – Final Results)

| Rank | Model                | Accuracy | Macro F1 | ROC-AUC (OvR) | Notes                             |
|------|----------------------|----------|----------|---------------|-----------------------------------|
| 1    | **Logistic Regression** | **0.9773** | **0.8641** | **0.9932** | Surprisingly strongest baseline  |
| 2    | XGBoost              | 0.9667   | 0.8161   | 0.9898        | Very close second                 |
| 3    | Random Forest        | 0.9172   | 0.7631   | 0.9804        | Good but clearly beaten           |

### Detailed Classification Reports 

#### 1. Logistic Regression (Best Baseline)

              precision    recall  f1-score   support

        High       0.53      0.76      0.62        62
         Low       0.99      0.99      0.99      1741
      Medium       0.99      0.97      0.98      1892
    accuracy                           0.98      3695


#### 2. XGBoost

              precision    recall  f1-score   support

        High       0.56      0.45      0.50        62
         Low       0.98      0.99      0.98      1741
      Medium       0.97      0.97      0.97      1892
    accuracy                           0.97      3695


#### 3. Random Forest

              precision    recall  f1-score   support

        High       0.51      0.39      0.44        62
         Low       0.90      0.96      0.93      1741
      Medium       0.94      0.89      0.92      1892
    accuracy                           0.92      3695
