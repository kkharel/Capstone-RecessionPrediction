### Project Title

**Author**

#### Executive summary

#### Rationale
Why should anyone care about this question?

#### Research Question
What are you trying to answer?

#### Data Sources
What data will you use to answer you question?

#### Methodology
What methods are you using to answer the question?

#### Results
The correlation analysis reveals the strength and direction of linear relationships between various economic indicators and the occurrence of a recession. As expected, the recession variable has a perfect correlation with itself (1.0). Among the positively correlated features, the unemployment rate (0.198) and various Treasury yields such as the 30-year (0.148), 10-year (0.146), and federal funds rate (0.134) show mild positive associations with recessions. This suggests that during or just before recessions, interest rates and unemployment may rise slightly, possibly reflecting tightening financial conditions or deteriorating labor markets.

Conversely, several variables exhibit moderate to strong negative correlations with recessions. Notably, consumer sentiment (-0.379) and the real GDP growth rate (-0.369) show the strongest negative correlations, indicating that consumer confidence and economic growth tend to decline significantly during recessionary periods. Other indicators such as building permits (-0.312), capacity utilization (-0.305), and housing starts (-0.287) also show moderate negative correlations, highlighting that construction activity and industrial usage shrink during economic downturns.

Market indices like the S&P 500 (-0.151) and Dow Jones Industrial Average (-0.148) are slightly negatively correlated with recessions, which aligns with the general decline in equity markets during economic contractions. Similarly, GDP (-0.118) and nonfarm payroll (-0.117) show weak negative correlations, reflecting reduced output and employment during recessions.

Overall, the analysis confirms that economic output, consumer behavior, and business activity generally decline during recessions, while unemployment and some interest rates show slight increases, consistent with typical recession dynamics.

The line plot of 10Y–2Y yield spread over time show periods where the yield spread fell below zero—indicated an inversion - frequently preceded or coincided with the onset of recessions. This visual alignment supports the historical reliability of yield curve inversions as an early warning signal for economic downturns.

The baseline logistic regression model with balanced class weight shows a strong fit on the training data with a high score of 0.98, but it struggles with generalizing to unseen data, as reflected in the lower test score of 0.91. This discrepancy suggests potential overfitting, where the model has learned the patterns of the training data well but fails to capture the minority class effectively on the test data. Specifically, for the target class "1.0" (recession), the model has perfect precision (1.00), indicating that when it predicts a recession, it is almost always correct. However, the recall for this class is very low (0.05), meaning the model misses most actual recession periods, leading to a high number of false negatives. This is also reflected in the low F1-score of 0.10 for the recession class, highlighting poor performance in predicting the minority class. On the other hand, the model performs well for the majority class "0.0" (no recession), with a high precision (0.91), perfect recall (1.00), and a strong F1-score (0.95). The macro average scores show the imbalance, with recall for the minority class being significantly lower, while the weighted average scores are skewed due to the dominance of the majority class. This indicates that the model is biased towards predicting no recession, leading to poor performance in identifying recession periods. 

The strongest predictor based on this model is consumer_sentiment_lag1, with a large negative coefficient (-1.36), indicating that a sharp decline in consumer confidence one month prior is a strong signal of an approaching recession. This is closely followed by unemployment_rate_lag2 (-1.22), suggesting that higher unemployment levels two months earlier are a strong precursor to economic contraction. Gold_price_lag3 (-1.04) and consumer_sentiment_lag2 (-1.00) also showed strong negative associations, reflecting both investor caution and sustained consumer pessimism over multiple periods. The third lag of unemployment rate (unemployment_rate_lag3, -0.94) further confirmed that deteriorating labor market conditions are robust early warnings. Interestingly, laborforce_participation_lag3 had a positive coefficient (0.76), potentially indicating temporary resilience in labor force engagement before broader economic impacts materialize. Gold_price_lag2 (-0.74), real_gdp_growth_rate_lag1 (-0.73), and capacity_utilization_lag1 (-0.71) all demonstrated negative contributions, highlighting that contraction in economic output, production efficiency, and investor flight to safety typically precede recessions. Lastly, building_permits_lag1 (-0.68) also negatively correlated with recession, underscoring the importance of a slowdown in housing activity as a leading indicator. Collectively, these variables provide comprehensive, lagged view of consumer, labor, investment, and industrial signals that meaningfully anticipate economic downturns.
#### Next steps
What suggestions do you have for next steps?
To improve performance, especially in predicting recessions, techniques like resampling, adjusting class weights, or threshold tuning could be explored. Additionally, experimenting with different models or further feature engineering might help capture more nuanced patterns in the data


#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information