# Interest Rate Pricing Experiment & Elasticity Simulation
Purpose - Simulate interest rates, predict default risk, model demand elasticity, and test the impact of rate discounts using the Home Credit Default Risk dataset.

Dataset:  
- Kaggle Home Credit Default Risk – application_train.csv  
- Original: 307,511 rows × 122 columns  
- Sampled: 75,000 rows for speed  
- Target: `TARGET` (1 = default, ~8% rate)

Key Features:
- Simulated interest rate proxy: `(AMT_ANNUITY / AMT_CREDIT) × 12 × 100`
- Risk model: XGBoost (AUC ~0.74) to predict default probability
- Demand proxy: Linear inverse (`1 / (1 + rate/100)`) – normalized 0–1
- A/B simulation: 10% rate discount → **+13.58% demand uplift**

Tech Stack:
- Python
- Pandas & NumPy
- Matplotlib & Seaborn (EDA & visualization)
- Scikit-learn (preprocessing, Logistic Regression, evaluation)
- XGBoost (risk prediction)
- Simulated A/B testing & elasticity modeling

Project Highlights:
- Data cleaning: Age/employment years, missing value imputation, flags
- Simulated rate EDA: Distribution, default rate by bin, external scores by rate
- Risk model comparison: Logistic (~0.73) vs XGBoost (~0.74)
- Elasticity curve: Strong negative demand response to higher rates
- A/B discount simulation: 10% lower rate → **+13.58% demand** (volume gain vs margin loss)

Business Insight:
Targeted rate discounts to low-risk borrowers (high external scores) can significantly increase demand and profit while controlling default risk.

How to Run:
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt` (pandas, matplotlib, seaborn, scikit-learn, xgboost)
3. Download dataset: https://www.kaggle.com/competitions/home-credit-default-risk/data
4. Run the notebook: `jupyter notebook interest_rate_simulation.ipynb`

**Dataset Source**:  
https://www.kaggle.com/competitions/home-credit-default-risk/data

Screenshots included in repo:
- Elasticity curve (demand proxy vs rate)
- A/B uplift bar plot
- Feature importance (risk model)

Full code, EDA, models, and simulation results inside!
