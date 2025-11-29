# Driver Risk Prediction Project

A comprehensive machine learning project analyzing driver risk through Supervised Learning, Unsupervised Learning, and Reinforcement Learning approaches.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Part 1: Data Preprocessing & Feature Engineering](#part-1-data-preprocessing--feature-engineering)
- [Part 2: Supervised Learning](#part-2-supervised-learning)
- [Part 3: Unsupervised Learning](#part-3-unsupervised-learning)
- [Part 4: Reinforcement Learning](#part-4-reinforcement-learning)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Project Overview

This project aims to predict and understand driver risk patterns using three different machine learning paradigms:

1. **Supervised Learning**: Classify drivers into risk categories (Low/Medium/High)
2. **Unsupervised Learning**: Discover hidden patterns in driver behavior without labels
3. **Reinforcement Learning**: Train an AI agent to learn optimal driving behavior

---

## Dataset

### Accident Dataset (df_accident)
- **Size**: 12,316 records
- **Features**: 14 categorical + 1 numerical (Accident_severity)
- **Key Features**:
  - Driver demographics (age, gender, education)
  - Environmental conditions (weather, road type, lighting)
  - Behavioral factors (cause of accident, driving experience)
  - Accident severity (0-2 scale)

### Driver Dataset (df_driver)
- **Size**: 840 records
- **Features**: 7 numerical + 7 categorical
- **Key Features**:
  - Traffic density, speed limit, number of vehicles
  - Driver age, experience, alcohol consumption
  - Road conditions, vehicle type
  - Binary accident indicator

---

## Project Structure

```
driver-risk-prediction/
├── data/
│   ├── cleaned.csv
│   ├── dataset_traffic_accident_prediction1.csv
│   └── df_accident
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_supervised_learning.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   └── 04_reinforcement_learning.ipynb
└── README.md
```

---

## Part 1: Data Preprocessing & Feature Engineering

### 1.1 Data Cleaning
- **Missing Values**: Filled using median (numerical) and mode (categorical)
- **One-Hot Encoding**: Applied to all 14 categorical features
- **Final Shape**: 12,316 × 100 features

### 1.2 Feature Engineering

Created three custom risk scores:

#### Driver Risk Score (0-1 scale)
Factors:
- Age band (young/elderly = higher risk)
- Driving experience (< 2 years = high risk)
- Education level (lower education = higher risk)
- Alcohol consumption (average driver dataset)

```python
driver_risk = (age_risk + experience_risk + education_risk + alcohol_risk) / 4
```

#### Environmental Risk Score (0-1 scale)
Factors:
- Weather conditions (rain/fog/snow = high risk)
- Road surface type (earth/gravel = high risk)
- Light conditions (darkness = high risk)
- Junction type (complex junctions = medium risk)

#### Behavioral Risk Score (0-1 scale)
Based on cause of accident:
- **High risk (1.0)**: No distancing, drunk driving, overtaking
- **Medium risk (0.6)**: High speed
- **Low risk (0.2)**: Following rules, minor violations

#### Total Risk Score & Classification
```python
total_risk = (driver_risk + env_risk + behaviour_risk) / 3

Risk Classes:
- Low:    score < 0.4
- Medium: 0.4 ≤ score < 0.7
- High:   score ≥ 0.7
```

**Final Distribution**:
- Medium: 6,307 (51.2%)
- Low: 5,802 (47.1%)
- High: 207 (1.7%)

---

## Part 2: Supervised Learning

### 2.1 Models Trained

Three classification models with hyperparameter tuning:

#### Random Forest
- Best params: `max_depth=None, min_samples_split=5, n_estimators=200`
- GridSearchCV best score: 95.64%

#### Logistic Regression
- Best params: `C=10, penalty='l2', solver='lbfgs'`
- Balanced class weights

#### XGBoost
- Best params: Optimized through GridSearchCV
- Multi-class classification (`objective='multi:softmax'`)

### 2.2 Data Handling

**Class Imbalance Solution**: SMOTE (Synthetic Minority Over-sampling)
- Original: High=145, Low=4,061, Medium=4,415
- After SMOTE: High=4,415, Low=4,415, Medium=4,415

**Train-Test Split**: 70-30 with stratification

### 2.3 Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Random Forest | ~95%+ | Best overall performance |
| Logistic Regression | ~85-90% | Fast, interpretable |
| XGBoost | ~93-95% | Strong performance, ensemble method |

### 2.4 Model Interpretability

**SHAP Analysis** applied to all models:
- Feature importance visualization
- Class-specific SHAP values
- Understanding prediction drivers

---

## Part 3: Unsupervised Learning

### 3.1 Methods Applied

#### K-Means Clustering
- Tested k=2 to k=10
- Optimal k: 2-3 (based on Silhouette Score)
- Silhouette Score: 0.09-0.12 (low, indicating overlapping clusters)

**Elbow Method Results**:
- No clear "elbow" point
- Gradual decrease in inertia

#### PCA (Principal Component Analysis)
- Components tested: 2 (visualization) and 57 (70% variance)
- First 2 components: Only 4.57% variance explained
- Challenge: One-hot encoded features are inherently high-dimensional

#### Hierarchical Clustering
- Method: Ward linkage
- Dendrogram visualization (1000 samples)
- Silhouette Score: 0.088 (similar to K-Means)

#### DBSCAN (Density-Based Clustering)
- Parameters: `eps=5, min_samples=50`
- Result: 70.5% outliers (8,683/12,316)
- Too sparse for density-based clustering

### 3.2 Cluster Validation

**Adjusted Rand Index (ARI)** vs. true risk_class:
- K-Means: 0.027
- Hierarchical: 0.0004
- Conclusion: Unsupervised patterns do not match supervised risk classes

### 3.3 Key Insights

**Why Clustering Failed**:
1. **Binary one-hot features**: Euclidean distance not ideal
2. **High dimensionality**: 99 features, sparse data
3. **Risk classes naturally overlap**: Low/Medium/High not well-separated in feature space
4. **Unsupervised finds different patterns**: Not aligned with human-defined risk labels

**This is expected**: Unsupervised learning discovers data-driven patterns, not necessarily matching domain-defined categories.

---

## Part 4: Reinforcement Learning

### 4.1 Environment Design

**Driving Environment** implemented as a Markov Decision Process (MDP):

#### State Space (8 dimensions)
Static:
- Age band, gender, education
- Weather conditions, road type

Dynamic:
- Current speed (km/h)
- Fatigue level (0-100)
- Distance to goal (km)

**Discrete State**: Reduced to 27 states (3×3×3) for Q-Learning:
- Speed: Low (0-40), Medium (41-80), High (81+)
- Fatigue: Low (0-30), Medium (31-60), High (61+)
- Distance: Close (0-30), Medium (31-70), Far (71+)

#### Action Space (3 actions)
0. **Careful Driving**: Speed -10, fatigue +1, safe
1. **Normal Driving**: Speed =60, fatigue +2, moderate
2. **Aggressive Driving**: Speed +20, fatigue +5, risky

#### Reward Function
- +10: Each safe step
- +100: Reach destination safely
- -100: Crash occurs
- -5: Penalty for aggressive driving

#### Episode Termination
- Crash occurs (negative outcome)
- Destination reached (positive outcome)
- Max 100 steps

#### Crash Probability Calculation
```python
crash_prob = base_risk (1%)
           + action_risk (0.5-3%)
           + fatigue_risk (fatigue/1000)
           + speed_risk ((speed-80)/1000)
Max probability capped at 50%
```

### 4.2 Q-Learning Agent

**Algorithm**: Q-Learning with epsilon-greedy exploration

**Hyperparameters**:
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.95
- Initial epsilon: 1.0
- Epsilon decay: 0.995
- Min epsilon: 0.01

**Q-Table**:
- Dictionary format: `{state: [Q(s,a0), Q(s,a1), Q(s,a2)]}`
- Final size: 14 states (out of 27 possible)
- Efficient learning: Only visited states stored

### 4.3 Training Process

**Setup**:
- Episodes: 1,000
- Max steps per episode: 100
- Different driver profile each episode (cycling through dataset)

**Training Progress**:
```
Episode 100:  Avg Reward = 135.85, Epsilon = 0.01, Success = 56.1%
Episode 500:  Avg Reward = 148.85, Epsilon = 0.01, Success = 73.4%
Episode 1000: Avg Reward = 109.20, Epsilon = 0.01, Success = 98.8%
```

**Final Training Results**:
- Total crashes: 490 (49%)
- Total successes: 510 (51%)
- Success rate: 50.2%
- Average reward: 110-176 (stable)

### 4.4 Testing Phase

**Test Setup**: 100 episodes, exploitation only (epsilon=0)

**Test Results**:
- Average reward: 99.40
- Success rate: 47.0%
- Crash rate: 53.0%
- Best reward: 250
- Worst reward: -100

**Conclusion**: No overfitting. Test performance matches training.

### 4.5 Learned Policy

**Q-Values Analysis** (Top states):

| State | Speed | Fatigue | Distance | Best Action | Q-Value |
|-------|-------|---------|----------|-------------|---------|
| (1,1,0) | Medium | Medium | Close | Careful | 92.2 |
| (0,1,0) | Low | Medium | Close | Normal | 91.2 |
| (2,0,1) | High | Low | Medium | Normal | 83.8 |
| (0,0,2) | Low | Low | Far | Aggressive | 63.2 |

**Policy Interpretation**:
1. Near destination + any fatigue → Drive carefully
2. High speed → Normalize speed
3. Far from goal + fresh → Can be aggressive
4. Low speed + far → Speed up

**Result**: Agent learned a safe and logical driving policy.

---

## Key Findings

### Overall Project Insights

1. **Supervised Learning**: Excellent performance
   - Random Forest achieved ~95% accuracy
   - SHAP analysis reveals key risk drivers
   - Feature engineering (risk scores) was effective

2. **Unsupervised Learning**: Limited success
   - Clustering struggled with binary features
   - PCA explained only 4.57% variance (2D)
   - Unsupervised patterns do not match domain-defined risk classes
   - This is normal: Different learning objectives

3. **Reinforcement Learning**: Successful learning
   - Agent learned logical, safe driving policy
   - 47-50% success rate (random = ~33%)
   - No overfitting, stable performance
   - Optimal behavior: Careful near goal, aggressive when far

### Comparative Analysis

| Approach | Goal | Success | Key Challenge |
|----------|------|---------|---------------|
| Supervised | Predict risk class | High accuracy | Class imbalance (solved with SMOTE) |
| Unsupervised | Find patterns | Low scores | High-dimensional binary features |
| Reinforcement | Learn optimal behavior | Logical policy | Environment design complexity |

---

## Requirements

```python
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.41.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
```

Install all requirements:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn scipy
```

---

## Usage

### 1. Data Preprocessing
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load data
df_accident = pd.read_csv('cleaned.csv')
df_driver = pd.read_csv('dataset_traffic_accident_prediction1.csv')

# Feature engineering (see Part 1)
# Run risk score calculations
# Save processed data
```

### 2. Supervised Learning
```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train Random Forest
rf = RandomForestClassifier(class_weight="balanced", n_estimators=200)
rf.fit(X_train_res, y_train_res)
```

### 3. Unsupervised Learning
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### 4. Reinforcement Learning
```python
# Create environment
env = DrivingEnvironment(driver_features=X.iloc[0])
state = env.reset()

# Train Q-Learning agent
agent = QLearningAgent()
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
```

---

## Visualization Examples

### Supervised Learning
- Confusion matrices for all models
- SHAP summary plots (feature importance)
- Classification reports

### Unsupervised Learning
- Elbow method & Silhouette score plots
- 2D PCA scatter plots (clusters & true labels)
- Hierarchical clustering dendrogram
- Q-Value heatmaps for top states

### Reinforcement Learning
- Episode rewards over time (with moving average)
- Episode lengths over time
- Success rate per 100 episodes
- Q-Value heatmap for learned policy

---

## Learning Outcomes

1. **Feature Engineering**: Created domain-specific risk scores
2. **Class Imbalance**: Solved with SMOTE oversampling
3. **Model Comparison**: Evaluated multiple ML algorithms
4. **Interpretability**: Used SHAP for explainable AI
5. **Clustering Challenges**: Understood limitations with binary features
6. **RL Environment Design**: Built custom MDP from scratch
7. **Q-Learning**: Implemented tabular reinforcement learning
8. **Policy Evaluation**: Validated learned behavior makes logical sense

---

## Future Improvements

### Supervised Learning
- Ensemble stacking (combine RF + XGBoost)
- More feature interactions
- Time-based validation (if temporal data available)

### Unsupervised Learning
- Try different distance metrics (Hamming, Jaccard for binary data)
- Mixed-type clustering (e.g., k-prototypes)
- Deep clustering with autoencoders

### Reinforcement Learning
- Deep Q-Network (DQN): Use neural networks instead of Q-table
- Policy Gradient: Try REINFORCE or Actor-Critic
- More complex environment: Multi-agent, dynamic traffic
- Real-world validation: Test on driving simulator

---

