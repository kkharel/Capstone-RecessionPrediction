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
        * Voting Classifier
    * Models are evaluated using F1-score (tuned with a custom threshold) as the primary metric, alongside precision, recall, accuracy, ROC AUC, and Average Precision.

#### Results

### Final Model Results Summary:

| Model                  | Best Threshold (Train) | F1 Score (Test, Tuned Threshold) | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | ROC AUC Score | Average Precision Score |
|:-----------------------|-----------------------:|---------------------------------:|---------------:|--------------:|----------:|-------:|---------:|--------------:|------------------------:|
| VotingClassifier       | 0.762                  | 0.789                            | 0.9857        | 0.9515       | 0.8333   | 0.75  | 0.789    | 0.9352       | 0.8123                  |
| Bagging_Classifier     | 0.644                  | 0.780                            | 0.94          | 0.9455       | 0.7619   | 0.80  | 0.780    | 0.9434       | 0.7460                  |
| XGBClassifier          | 0.871                  | 0.762                            | 0.9857        | 0.9394       | 0.7273   | 0.80  | 0.762    | 0.9193       | 0.7940                  |
| SVC                    | 0.673                  | 0.737                            | 0.9857        | 0.9394       | 0.7778   | 0.70  | 0.737    | 0.9479       | 0.8108                  |
| RandomForestClassifier | 0.634                  | 0.732                            | 0.9457        | 0.9333       | 0.7143   | 0.75  | 0.732    | 0.9317       | 0.7857                  |
| LogisticRegression     | 0.614                  | 0.604                            | 0.9571        | 0.8727       | 0.4848   | 0.80  | 0.604    | 0.9141       | 0.5609                  |

The ensemble VotingClassifier achieved the best overall performance with a tuned threshold F1 score of 0.789 on the test set, paired with strong precision (0.83) and recall (0.75), indicating a balanced ability to correctly identify positive cases while limiting false positives. Bagging and XGBoost classifiers also performed well, with F1 scores close to 0.78 and 0.76 respectively, showing strong generalization and competitive recall values around 0.8.

SVC and RandomForest classifiers followed closely, providing solid accuracy and AUC scores above 0.93, with SVC standing out with the highest ROC AUC (0.95), suggesting excellent discrimination capability. Logistic Regression lagged behind in F1 and precision despite good recall, indicating it tends to flag more positives but with lower precision.

Overall, the results suggest that ensemble methods and boosted trees outperform simpler linear models in this classification task, especially when threshold tuning is applied to optimize F1 score on the test set.


### Feature Importances for Each Model

**Model: LogisticRegression**

| Feature                                | Importance |
| :-------------------------------------| ----------:|
| SP500_price_roll_mean6                 | 0.522601   |
| core_CPI_roll_mean3                    | 0.475610   |
| core_CPI                              | 0.457520   |
| industrial_production_roll_mean6      | 0.450118   |
| consumer_sentiment_roll_mean6         | 0.203920   |
| nonfarm_payroll                      | 0.202524   |
| housing_start                        | 0.170260   |
| real_GDP_growth_rate                 | 0.139051   |
| nonfarm_payroll_roll_mean3            | 0.078187   |
| median_sale_price_new_houses_roll_mean6 | 0.059299 |

---

**Model: RandomForestClassifier**

| Feature                            | Importance |
| :---------------------------------| ----------:|
| nonfarm_payroll_roll_mean3         | 0.204649   |
| real_GDP_growth_rate_roll_mean6    | 0.198685   |
| core_CPI_roll_mean6                | 0.171489   |
| nonfarm_payroll                   | 0.146579   |
| industrial_production_roll_mean6  | 0.085590   |
| CPI_roll_mean6                   | 0.056688   |
| housing_start                   | 0.054814   |
| SP500_price_roll_mean6           | 0.050661   |
| nonfarm_payroll_roll_mean6        | 0.022469   |
| 1Y_treasury_yield_roll_std6       | 0.008376   |

---

**Model: Bagging_Classifier**

| Feature                            | Importance |
| :---------------------------------| ----------:|
| nonfarm_payroll_roll_mean3         | 0.276180   |
| nonfarm_payroll                  | 0.183783   |
| real_GDP_growth_rate_roll_mean6   | 0.173631   |
| industrial_production_roll_mean6  | 0.095152   |
| core_CPI_roll_mean6              | 0.083564   |
| SP500_price_roll_mean6           | 0.067149   |
| CPI_roll_mean6                 | 0.035630   |
| core_CPI_roll_mean3            | 0.034292   |
| nonfarm_payroll_roll_mean6        | 0.029871   |
| housing_start                  | 0.020749   |

---

**Model: XGBClassifier**

| Feature                                         | Importance |
| :----------------------------------------------| ----------:|
| 3Y_treasury_yield                              | 0.070186   |
| nonfarm_payroll_roll_mean3                      | 0.049318   |
| industrial_production_roll_mean6                | 0.031649   |
| real_GDP_growth_rate_roll_mean6                 | 0.030209   |
| yield_spread_10Y_2Y_lag6                        | 0.021241   |
| nonfarm_payroll                                | 0.018815   |
| housing_start_lag3                             | 0.013586   |
| nonfarm_business_sector_real_output_per_hour_roll_std6 | 0.012793   |
| industrial_production_roll_mean3                | 0.011660   |
| median_sale_price_new_houses_lag3               | 0.011552   |

---

**Model: SVC**

This model does not provide feature importances or coefficients.


The results above show a consistent importance of labor market indicators such as nonfarm payrolls and economic measures like GDP growth and core CPI across different models. The Logistic Regression highlights market price and inflation indicators as strongly predictive, while tree-based models also emphasize payroll and production metrics. Notably, SVC does not provide feature importances, reflecting the model's nature

### Model Limitations:
The models for SVC do not provide direct feature importances, limiting interpretability for these specific models.
The effectiveness of the models is heavily reliant on the quality and availability of economic data, and real-time data might differ from historical values used for training.
While the F1-score is high, the rare nature of recessions means that even a small number of false positives or false negatives can have significant implications.

#### Next Steps

* **Further Feature Engineering**: Explore additional leading economic indicators or composite indices.
* **Advanced Time Series Models**: Investigate deep learning models like LSTMs or Transformers, which are well-suited for sequence data.
* **Real-time Monitoring**: Develop a system for continuous data ingestion and model inference to provide real-time recession probabilities.
* **Sensitivity Analysis**: Conduct more thorough sensitivity analyses to different parameter choices or data assumptions.
* **User-Selected Forecast**: Horizon: Add interactive date pickers to allow users to select custom start and end dates for the input data, as well as adjust the forecast horizon dynamically. This will enable tailored recession predictions for specific future periods and improve the app’s flexibility and usability.
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
* [`recession_predictor_app.py`](recession_predictor_app.py): Contains minimal viable product application.
* 
- [Link to notebook](https://github.com/kkharel/Capstone-RecessionPrediction/blob/main/RecessionPrediction.ipynb)
- [Link to App](https://capstone-recessionprediction-j3jew7zh5gknh9r8vk73s5.streamlit.app/)
  
For project_config file, please provide these information: API_KEY = "FREDAPI Key", TARGET = "recession", PREDICTION_HORIZON_MONTHS = 1, MAIN_TRAIN_TEST_SPLIT_DATE = datetime(2007, 1, 1), N_ITER_SEARCH = 100, RANDOM_STATE = 42

##### Contact and Further Information

For any questions or further information, please contact kushalkharelsearch@gmail.com or connect on [linkedin](https://www.linkedin.com/in/kushalkharel/).
