# Fusemachines AI Fellowship – Weekly Progress Log

Welcome to my 24-week journey through the Fusemachines AI Fellowship. This repository serves as a public log of my learning progress, challenges, and hands-on projects. Each week includes a short summary, key takeaways, and links to related code or notes.

> **Why this Fellowship?**  
> I joined this program to sharpen my skills in AI/ML, build real-world projects, and grow with a learning community. This log helps me stay accountable and reflect on my growth.

---

## Table of Contents

- [Weekly Learning Logs](#weekly-learning-logs)  
  - [Week 1: Introduction to AI/ML](#week-1-introduction-to-aiml)  
  - [Week 2: 12-Factor App for Machine Learning Systems](#week-2-12-factor-app-for-machine-learning-systems)  
  - [Week 3: Data Wrangling: Pandas and SQL](#week-3-data-wrangling-pandas-and-sql)  
  - [Week 4: Data Visualization and Presentation](#week-4-data-visualization-and-presentation)  
  - [Week 5: Linear Models](#week-5-linear-models)
  - [Week_6: Beyond Linear Models](#week-6-beyond-linear-models)
  - [Week_7: Ensemble Learning and Optimization Strategies](#week-7-ensemble-learning-and-optimization-strategies)
  - [Week_8: Feature Engineering and ML Pipelines](#week-8-feature-engineering-and-ml-pipelines)
  - [Week 9: Time Series Forecasting](#week-9-time-series-forecasting)
  - [Week 10: Unsupervised ML Techniques](#week-10-unsupervised-ml-techniques)
    
    
  

- [LinkedIn Recaps](#linkedin-recaps)

---

## Weekly Learning Logs

### Week 1: Introduction to AI/ML

**Pre-Session Prep**

- Intermediate Python: Lists, Tuples, Dictionaries, Sets, Strings, Collections, Itertools, Lambda, Exceptions, Logging, JSON, Decorators, Generators, Threading & Multiprocessing  
- Mathematics: Linear Algebra, Matrix Calculus, Probability Theory  
- AI/ML: Applications, typical ML workflow

**Live Session Recap**

- Discussed AI evolution, real-world impact, and fellowship roadmap

**Post-session Work**

- Created a personalized learning plan and defined focus areas

**Key Insight**  
> I learned that setting clear goals early helped me stay focused and intentional throughout the program.

---

### Week 2: 12-Factor App for Machine Learning Systems

**Pre-Session Prep**

- Git basics, project templates with Cookiecutter, and Python ML libraries  
- Topics included:
  - 12-Factor App methodology  
  - REST APIs with FastAPI  
  - Async programming  
  - Logging and debugging  
  - Docker containerization

**Live Session Recap**

- Applied 12-Factor principles to real ML system design  
- Explored deployment-ready architecture and development best practices

**Post-session Work**

I built a FastAPI microservice applying as many 12-Factor principles as possible, focusing on structure, clarity, and portability.

> **Implementation Details:**  
[GitHub – Explore Cafe API](https://github.com/KushalRegmi61/Explore-Cafe-API)

**Key Insight**  
> I realized that engineering best practices are essential to turn ML projects into reliable, scalable systems.

---

### Week 3: Data Wrangling Pandas and SQL

**Pre-Session Prep**

- Pandas basics and SQL queries: filtering, joins, aggregation  
- Data validation with Pydantic and real-world data cleaning scenarios

**Live Session Recap**

- Worked through hands-on challenges integrating Pandas and SQL for real-world data analysis

**Post-session Work**

- Completed 20 SQL queries involving customers, employees, subqueries, ranking, and data updates  
- In Pandas: created/loaded datasets, cleaned and transformed them, and ran exploratory analysis on product ratings

> **Implementation Details:**  
[GitHub – Data Wrangling with SQL and Pandas](https://github.com/KushalRegmi61/Data_Wrangling_with_SQL_and_Pandas/tree/master)


**Key Insight** 
> I found that combining SQL and Pandas made data cleaning and analysis much more efficient and flexible.

---

### Week 4: Data Visualization and Presentation

**Pre-Session Prep**

- Studied visualization types, chart selection, layout, typography, and ethical design  
- Explored libraries including Altair, Matplotlib, Plotly, and Seaborn

**Live Session Recap**

- Practiced exploratory vs explanatory visualization techniques  
- Built and evaluated charts for 1D, 2D, and multi-dimensional datasets

**Post-session Work**

- Analyzed and visualized the Seaborn Tips dataset with exploratory and explanatory approaches  
- Focused on clarity, relevance, and ethical presentation of data insights

> **Implementation Details:**  
[GitHub – Data Visualization](https://github.com/KushalRegmi61/data_visualization/tree/master)

**Key Insight** 
> I learned how effective visual storytelling can make insights clearer and more impactful.

---

### Week 5: Linear Models

**Pre-Session Prep**

- Linear and polynomial regression  
- Performance metrics: R², RMSE  
- MLE and least squares  
- Regularization: Lasso, Ridge, ElasticNet  
- Logistic regression: binary, multiclass (OvR, OvO)  
- Optimization: gradient descent for simple and multiple models

**Live Session Recap**

- Implemented real-world linear model use cases  
- Interpreted model coefficients and compared regularized vs non-regularized models

**Post-session Work**

Using a student performance dataset:

1. **Regression Modeling:** Predicted final grades (G3) using linear regression, evaluated with R² and RMSE  
2. **Classification Task:** Predicted pass/fail outcomes with logistic regression using accuracy, precision, recall, and F1-score

Key steps:
- Feature selection using correlation heatmaps  
- Categorical encoding  
- Performance benchmarking with dummy classifiers

> **Implementation Details:**  
[GitHub – Linear Models](https://github.com/KushalRegmi61/Linear_Models/tree/master)

**Key Insight**  
> I saw how linear models, despite their simplicity, offer both predictive power and interpretability.

---

### Week 6: Beyond Linear Models

This week focused on **Beyond Linear Models: Discriminative and Generative Techniques**, covering non-linear, interpretable classifiers and probabilistic methods beyond the limitations of linear decision boundaries.

**Pre-Session Prep**

* Decision Trees: impurity metrics (Gini, Entropy), continuous variable handling, pruning, early stopping
* K-Nearest Neighbors (KNN): classification, regression, distance metrics (Brute Force, KD-Tree, Ball Tree)
* Naive Bayes: conditional independence, probabilistic modeling, sentiment analysis
* Support Vector Machines (SVM): margin maximization, kernel methods (linear, RBF, polynomial), C-SVM, ν-SVM

**Live Session Recap**

* Compared tree-based models with linear baselines
* Demonstrated pruning and interpretability techniques

**Post-session Work**

Using a hotel booking dataset:

1. **Classification Task:** Predicted booking cancellations (`is_canceled`) with a Decision Tree Classifier

   * Evaluated using accuracy, confusion matrix, and classification report

Key steps:

* Imputing missing variables
* Label encoding of categorical variables
* Hyperparameter tuning (`max_depth`, `min_samples_split`)
* Early stopping to prevent overfitting
* Comparison of full vs shallow tree performance

> **Detailed Repo For Week_6:** <br>
> [Beyond Linear Models](https://github.com/KushalRegmi61/Beyond_Linear_Models)

**Key Insight**

> I explored how models like SVM, KNN, Naive Bayes, and Decision Trees offer flexible, interpretable, and often more effective alternatives to linear models across diverse tasks.
---


### Week 7: Ensemble Learning and Optimization Strategies

**Overview**
Covered ensemble learning techniques—Bagging, Boosting (AdaBoost, Gradient Boosting, XGBoost)—to improve model accuracy and generalization. Also focused on hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`.

**Pre-Session Prep**

* Reviewed bias–variance decomposition and its implications
* Studied Bagging, Random Forest, AdaBoost, Gradient Boosting, and XGBoost
* Covered model tuning strategies and validation techniques

**Live Session Recap**

* Compared Bagging vs Boosting approaches
* Discussed XGBoost internals: second-order optimization, regularization, shrinkage
* Demonstrated tuning with scikit-learn pipelines

**Post-Session Work**

* Implemented Decision Tree, Random Forest, AdaBoost, GradientBoost, and XGBoost
* Applied `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning
* Evaluated and compared performance improvements using ensemble methods
  
> **Detailed Repo For Week_7:** <br>
>[Ensemble Learning and Optimization Strategies](https://github.com/KushalRegmi61/ensemble-learning-and-optimization-strategies)

**Key Insight**
Ensemble learning reduces bias and variance. When combined with proper tuning, it leads to scalable, accurate, and production-ready models.

---


### Week 8: Feature Engineering and ML Pipelines

**Overview**
Focused on designing high-quality features and building reusable ML pipelines. Covered techniques for transforming, selecting, and extracting features from various data types (text, images, time), handling outliers and imbalanced datasets, and implementing robust workflows using scikit-learn pipelines.

**Pre-Session Prep**

* Introduced the importance of feature engineering and types of features (numerical, categorical, datetime, text, image)
* Explored encoding methods (OHE, label, target), scaling techniques (standard, robust, min-max), and mathematical transformations
* Studied feature selection methods: Filter (chi2, ANOVA), Wrapper (forward/backward), and Embedded (Lasso, tree-based)

**Live Session Recap**

* Demonstrated feature extraction using regex (text) and datetime processing
* Covered outlier detection techniques (Z-score, IQR, DBSCAN, Isolation Forest) and imbalance handling methods (SMOTE, cost-sensitive learning)
* Explained ML workflow structuring using `Pipeline()`, `FunctionTransformer()`, and `ColumnTransformer()`

**Post-Session Work**

* Created features from structured, textual, and image data using domain-specific transformations
* Implemented pipelines to chain encoding, scaling, and model training using scikit-learn
* Balanced imbalanced datasets using hybrid sampling methods (SMOTE+ENN) and built cost-sensitive classifiers

> **Detailed Repo For Week\_8:** <br>
> [Feature Engineering and ML Pipelines](https://github.com/KushalRegmi61/feature-engineering-and-ml-pipelines)

**Key Insight**
Good features often matter more than the model itself. Structured pipelines not only ensure clean workflows but also simplify reproducibility and real-world deployment.

---

### Week 9: Time Series Forecasting

**Overview**
Focused on classical time series forecasting using univariate AQI data. Covered decomposition, stationarity checks, model identification via ACF/PACF, and model evaluation with time-aware validation strategies.

**Pre-Session Prep**

* Introduced time series concepts: trend, seasonality, cyclicality, and noise
* Explained additive vs multiplicative decomposition
* Covered stationarity, transformations (log, power, differencing), and ADF test
* Explored ACF and PACF for ARIMA parameter selection
* Discussed univariate vs multivariate forecasting models

**Live Session Recap**

* Demonstrated resampling and seasonal-trend decomposition
* Applied transformations and differencing to achieve stationarity
* Visualized ACF/PACF to identify ARIMA terms
* Covered models: Naive, Holt-Winters, and ARIMA
* Explained time series cross-validation: expanding vs sliding windows
* Evaluated forecasts using MAE, RMSE, and residuals

**Post-session Work**

* Built a 7-day AQI forecast system for Kathmandu using ARIMA
* Interpolated missing values, ensured stationarity, and applied ADF test
* Selected best ARIMA model (ARIMA(1,1,1)) via MAE, MAPE
* Re-trained on full dataset to forecast AQI for July 24–30, 2025

> **Detailed Repo For Week\_9:** <br>
> [Kathmandu AQI Forecasting](https://github.com/KushalRegmi61/7-days-aqi-forecast-ktm)

**Key Insight**
Classical time series models like ARIMA can provide strong baseline forecasts if stationarity is ensured and seasonality is well-understood.

---

### Week 10: Unsupervised ML Techniques

**Overview**  
Focused on extracting insights from unlabeled data using clustering techniques and association rule mining. Covered KMeans, hierarchical clustering, K-Prototypes, and market basket analysis for customer behavior understanding.

**Pre-Session Prep**

* Explored differences between supervised vs. unsupervised learning
* Covered types of clustering: partition-based, hierarchical, density-based
* Introduced KMeans algorithm and its assumptions
* Studied linkage types and Lance-Williams formula in hierarchical clustering
* Discussed K-Prototypes for mixed-type data
* Introduced association rule mining: support, confidence, lift

**Live Session Recap**

* Implemented KMeans and evaluated clusters using inertia and silhouette scores
* Visualized dendrograms and used agglomerative clustering for hierarchy extraction
* Applied K-Prototypes to handle categorical and numerical data jointly
* Performed market basket analysis using Apriori and `association_rules`
* Discussed customer segmentation, anomaly detection, and recommendation use cases

**Post-session Work**

* Scaled numerical features (Age, Credit amount, Duration) after outlier handling
* Ran KMeans clustering, selected optimal `k` using elbow + silhouette methods
* Used linkage matrix to build and interpret hierarchical clusters
* Clustered full dataset with KPrototypes after imputing missing categorical values
* Applied association rule mining on a shopping mall dataset to uncover frequent co-purchased items and customer buying patterns

> **Detailed Repo For Week\_10:** <br>
> [Unsupervised Learning & Market Basket Analysis](https://github.com/KushalRegmi61/unsupervised-learning)

**Key Insight**  
Unsupervised learning reveals structure in data without labels. Clustering supports segmentation, while association rules uncover hidden item relationships both essential for actionable business insights.

---


## LinkedIn Recaps

- [Week 1–4 Recap](https://www.linkedin.com/posts/kushal-regmi-0b88a42aa_aifellowshipfusemachinesreadmemd-at-master-activity-7339869109536387073-0Zrf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEqGxYwBvISQU0D0hQ4gElKpiVYsO41o6NM)

- [Week 5 Post](https://www.linkedin.com/posts/kushal-regmi-0b88a42aa_github-kushalregmi61linearmodels-activity-7340702306566365185-QTCr?utm_source=share&utm_medium=member_android&rcm=ACoAAEqGxYwBvISQU0D0hQ4gElKpiVYsO41o6NM)
- [Week 6 Post](https://www.linkedin.com/posts/kushal-regmi-0b88a42aa_github-kushalregmi61beyondlinearmodels-activity-7342887916991340546-2iUb?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEqGxYwBvISQU0D0hQ4gElKpiVYsO41o6NM)
- [Week_7_Post](https://www.linkedin.com/posts/kushal-regmi-0b88a42aa_github-kushalregmi61essemble-learning-and-optimization-strategies-activity-7345409528735182849-Ynic?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEqGxYwBvISQU0D0hQ4gElKpiVYsO41o6NM)
- [Week_8_Post](https://www.linkedin.com/posts/kushal-regmi-0b88a42aa_github-kushalregmi61feature-engineering-and-ml-pipelines-activity-7350886257935192065-78YA?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEqGxYwBvISQU0D0hQ4gElKpiVYsO41o6NM)
