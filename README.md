# Predicting U.S. Economic Recessions with Machine Learning
---

**Author:** Kushal Kharel

---

#### Executive Summary

This project develops and evaluates a suite of machine learning models to predict U.S. economic recessions. Leveraging a comprehensive dataset of macroeconomic indicators and financial market data, the system performs rigorous time series preprocessing, feature engineering, and employs advanced cross-validation techniques to build robust predictive models. The primary goal is to provide timely and accurate recession forecasts, which can be invaluable for policymakers, businesses, and investors in navigating economic cycles.

#### Rationale

Recessions, characterized by significant declines in economic activity, have profound impacts on employment, financial markets, and overall societal well-being. Early and accurate prediction of these downturns allows for proactive measures, such as monetary and fiscal policy adjustments, business contingency planning, and investment strategy re-evaluation. By building a reliable predictive model, this project aims to contribute to economic stability and informed decision-making, mitigating the adverse effects of unexpected economic contractions.

#### Research Question

Can machine learning models, trained on preprocessed macroeconomic and financial time series data with appropriate time-series cross-validation, accurately predict the onset of U.S. economic recessions several months in advance?

#### Data Sources

The project utilizes a diverse set of time series data from various sources:

* **Federal Reserve Economic Data (FRED) API**:
    * Industrial Production, Capacity Utilization, Real Gross Domestic Product
    * Unemployment Rate, Total Nonfarm Employment, Labor Force Participation, Civilian Labor Force
    * CPI (Inflation), Core CPI, PPI (Wholesale Inflation)
    * Federal Funds Rate, Real M1/M2 Money Supply
    * Treasury Yields (10-Year, 5-Year, 7-Year, 30-Year, 3-Year, 2-Year, 1-Year)
    * Personal Consumption Expenditures, Real Disposable Personal Income, Total Consumer Credit, Trade Balance, Consumer Sentiment, Building Permits, Housing Starts, Median Sale Price New Houses, Nonfarm Business Sector Real Output Per Hour.
* **External CSVs**:
    * `gold.csv`: Historical Gold prices.
    * `sp500.csv`: Historical S&P 500 index data.
    * `dji.csv`: Historical Dow Jones Industrial Average (DJI) data.

#### Methodology

The methodology encompasses a multi-stage pipeline designed specifically for time series forecasting:

1.  **Data Pulling & Cleaning**: Economic indicators are fetched from the FRED API. This data is then merged with financial market data (Gold, S&P 500, DJI). Initial cleaning involves handling missing values (e.g., forward-filling quarterly columns) and standardizing date formats.
2.  **Time Series Preprocessing**:
    * **Growth Detection**: Identifies features exhibiting exponential growth using correlation with time and variance checks.
    * **Log Transformation**: Applies natural logarithm to features identified with exponential growth to stabilize variance and linearize trends.
    * **Stationarity Enforcement**: Uses the Augmented Dickey-Fuller (ADF) test to check for stationarity. Non-stationary series are differenced (first or second order) to achieve stationarity, which is crucial for many time series models and statistical properties.
3.  **Feature Engineering**: A comprehensive set of new features are created to capture various economic dynamics:
    * **Lagged Features**: Past values of key indicators (e.g., 3 and 6-month lags) to capture historical trends.
    * **Rolling Statistics**: Rolling means and standard deviations over various windows (e.g., 3 and 6 months) to reflect short-term trends, long-term trends, and volatility.
    * **Yield Spreads**: Calculation of the difference between 10-Year and 2-Year Treasury yields, a well-known predictor of recessions.
4.  **Data Preparation**: The processed and engineered data is then split chronologically into training and testing sets to ensure that the model is always evaluated on future, unseen data. The target variable (`recession`) is engineered to reflect future recessionary periods.
5.  **Model Training & Hyperparameter Tuning**:
    * A custom time-series cross-validation strategy is employed, where training is done on data up to a certain point, and validation is performed on subsequent, specific recessionary periods. This simulates real-world forecasting.
    * **SMOTE (Synthetic Minority Over-sampling Technique)** is integrated into `imblearn.pipeline` to address class imbalance, a common issue in recession prediction (recessions are rare events).
    * A range of classification models are trained and their hyperparameters are tuned using `BayesSearchCV` for efficient optimization:
        * Logistic Regression
        * Random Forest Classifier
        * Bagging Classifier (with Decision Tree as base estimator)
        * XGBoost Classifier
        * Support Vector Classifier (SVC)
        * K-Nearest Neighbors Classifier
        * Voting Classifier
    * Models are evaluated using F1-score (tuned with a custom threshold) as the primary metric, alongside precision, recall, accuracy, ROC AUC, and Average Precision.

#### Results

### Final Model Results Summary:

| model                  | best_threshold_tuned_on_train | f1_score_on_test_with_tuned_threshold | train_accuracy | test_accuracy | precision | recall | f1_score | roc_auc_score | average_precision_score |
| :--------------------- | :---------------------------- | :------------------------------------ | :------------- | :------------ | :-------- | :----- | :------- | :------------ | :---------------------- |
| XGBClassifier          | 0.782121                      | 0.864865                              | 0.982857       | 0.976744      | 0.941176  | 0.80   | 0.864865 | 0.946410      | 0.863192                |
| Bagging_Classifier     | 0.772222                      | 0.842105                              | 0.971429       | 0.972093      | 0.888889  | 0.80   | 0.842105 | 0.929744      | 0.864102                |
| VotingClassifier       | 0.801919                      | 0.829268                              | 0.985714       | 0.967442      | 0.809524  | 0.85   | 0.829268 | 0.975641      | 0.911291                |
| RandomForestClassifier | 0.693030                      | 0.820513                              | 0.977143       | 0.967442      | 0.842105  | 0.80   | 0.820513 | 0.929615      | 0.771769                |
| KNeighborsClassifier   | 0.722727                      | 0.782609                              | 0.985714       | 0.953488      | 0.782609  | 0.782609 | 0.782609 | 0.985714      | 0.782609                |
| SVC                    | 0.603939                      | 0.765957                              | 0.977143       | 0.948837      | 0.666667  | 0.90   | 0.765957 | 0.961410      | 0.734703                |
| LogisticRegression     | 0.762323                      | 0.711111                              | 0.974286       | 0.939535      | 0.640000  | 0.80   | 0.711111 | 0.939231      | 0.548671                |
### Individual Model Feature Importances/Coefficients

**Model: XGBoost**

| Feature                               | Importance |
| :------------------------------------ | :--------- |
| num__unemployment_rate_roll_mean6     | 0.057238   |
| num__DJI_price_roll_std3              | 0.051313   |
| num__civilian_labor_force_lag3        | 0.046314   |
| num__real_GDP_growth_rate_roll_mean6  | 0.038573   |
| num__nonfarm_payroll_roll_mean3       | 0.038207   |
| num__nonfarm_payroll                  | 0.037969   |
| num__housing_start_roll_mean3         | 0.036161   |
| num__housing_start_roll_mean6         | 0.032183   |
| num__real_GDP_growth_rate_lag3        | 0.032083   |
| num__nonfarm_payroll_roll_mean6       | 0.029872   |

**Model: RandomForest**

| Feature                             | Importance |
| :---------------------------------- | :--------- |
| num__real_GDP_growth_rate_roll_mean6 | 0.420222   |
| num__nonfarm_payroll                | 0.352276   |
| num__nonfarm_payroll_roll_mean6     | 0.070270   |
| num__DJI_price_roll_mean6           | 0.018286   |
| num__unemployment_rate_roll_mean3   | 0.017429   |
| num__SP500_price_roll_mean6         | 0.016505   |
| num__housing_start                  | 0.015089   |
| num__housing_start_roll_mean6       | 0.014025   |
| num__yield_spread_10Y_2Y_roll_std6  | 0.012323   |
| num__10Y_treasury_yield_roll_std6   | 0.012124   |

**Model: Bagging Classifier**
Model bag does not have direct feature importances or coefficients.

**Model: SVC**
Model svc does not have direct feature importances or coefficients.

**Model: K-Neighbors Classifier**
Model knn does not have direct feature importances or coefficients.

**Model: Logistic Regression (Absolute Coefficients - Top 10)**

| Feature                               | Coefficient |
| :------------------------------------ | :---------- |
| num__nonfarm_payroll                  | 0.036610    |
| num__real_GDP_growth_rate             | 0.029958    |
| num__unemployment_rate_roll_mean3     | 0.028566    |
| num__SP500_price_roll_mean6           | 0.028193    |
| num__housing_start                    | 0.027749    |
| num__unemployment_rate                | 0.027481    |
| num__consumer_sentiment_roll_mean6    | 0.027446    |
| num__nonfarm_payroll_roll_mean3       | 0.026956    |
| num__industrial_production            | 0.026115    |
| num__real_GDP_growth_rate_roll_mean6  | 0.025924    |

### Aggregated Feature Importance Across Models (Mean of Normalized Scores)

| Feature                               | Importance |
| :------------------------------------ | :--------- |
| num__real_GDP_growth_rate_roll_mean6  | 0.156834   |
| num__nonfarm_payroll                  | 0.135592   |
| num__nonfarm_payroll_roll_mean6       | 0.036652   |
| num__unemployment_rate_roll_mean6     | 0.022702   |
| num__housing_start_roll_mean6         | 0.018556   |
| num__unemployment_rate_roll_mean3     | 0.018316   |
| num__DJI_price_roll_std3              | 0.017829   |
| num__nonfarm_payroll_roll_mean3       | 0.016793   |
| num__civilian_labor_force_lag3        | 0.016292   |
| num__housing_start_roll_mean3         | 0.015614   |
The models demonstrated strong predictive capabilities, particularly the **XGBoost Classifier** and the **VotingClassifier**, which consistently achieved the highest F1-scores on the unseen test set after threshold tuning. Feature importance analysis revealed that indicators such as **Real GDP Growth Rate (6-month rolling mean)**, **Nonfarm Payroll**, and **Unemployment Rate (rolling means)** were the most influential predictors, aligning with economic theory. The custom time-series cross-validation and SMOTE proved effective in building robust models capable of identifying rare recession events.

#### Next Steps

* **Further Feature Engineering**: Explore additional leading economic indicators or composite indices.
* **Advanced Time Series Models**: Investigate deep learning models like LSTMs or Transformers, which are well-suited for sequence data.
* **Ensemble Optimization**: Further refine the `VotingClassifier` by experimenting with different base estimators or ensemble weights.
* **Real-time Monitoring**: Develop a system for continuous data ingestion and model inference to provide real-time recession probabilities.
* **Sensitivity Analysis**: Conduct more thorough sensitivity analyses to different parameter choices or data assumptions.

#### Outline of Project

* [`data_pull.py`](data_pull.py): Fetches raw economic data from the FRED API.
* [`data_cleaning.py`](data_cleaning.py): Cleans and merges economic data with financial market data.
* [`data_viz.py`](data_viz.py): Provides functions for data visualization (distributions, correlations, recession timelines).
* [`growth_detection.py`](growth_detection.py): Identifies features exhibiting exponential growth.
* [`log_transform.py`](log_transform.py): Applies log transformation to features exhibiting exponential growth.
* [`make_stationary.py`](make_stationary.py): Ensures time series stationarity using ADF test and differencing.
* [`baseline_model.py`](baseline_model.py): Trains and evaluates a baseline Logistic Regression model.
* [`feature_engineering.py`](feature_engineering.py): Generates lagged values, rolling statistics, and yield spread features.
* [`data_preparation.py`](data_preparation.py): Orchestrates data preprocessing and performs chronological train-test splits.
* [`custom_cv.py`](custom_cv.py): Implements a custom time series cross-validation strategy.
* [`model_training.py`](model_training.py): Contains pipelines and search spaces for various ML models and orchestrates their training and hyperparameter tuning.
* [`preprocessor.py`](preprocessor.py): Defines the preprocessing steps for the ML pipeline.
* [`project_config.py`](project_config.py): Stores global configuration variables like API keys, target column, random states, etc.
* [`utils.py`](utils.py): Contains utility functions for metrics, plotting, and feature importance summarization.

- [Link to notebook](https://github.com/kkharel/Capstone-RecessionPrediction/blob/main/RecessionPrediction.ipynb)

For project_config file, please provide these information: API_KEY = "FREDAPI Key", TARGET = "recession", PREDICTION_HORIZON_MONTHS = 1, MAIN_TRAIN_TEST_SPLIT_DATE = datetime(2007, 1, 1), N_ITER_SEARCH = 100, RANDOM_STATE = 42

##### Contact and Further Information

For any questions or further information, please contact kushalkharelsearch@gmail.com or connect on [linkedin](https://www.linkedin.com/in/kushalkharel/).
