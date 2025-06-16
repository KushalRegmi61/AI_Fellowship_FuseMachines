# 🤖 Fusemachines AI Fellowship – Weekly Progress

Welcome to my 24-week journey through the Fusemachines AI Fellowship. This repository serves as a public log of my learning progress, challenges, and hands-on projects. Each week includes a short summary, key takeaways, and links to related code or notes.

> 🎯 **Why this Fellowship?**  
> I joined this program to sharpen my skills in AI/ML, build real-world projects, and grow with a learning community. This log helps me stay accountable and reflect on my growth.

---

## 📚 Table of Contents

‣ [📘 Detailed Weekly Logs](#-detailed-weekly-logs)  
&nbsp;&nbsp;&nbsp;&nbsp;‣ [Week 1: Introduction to AI/ML](#-week-1-introduction-to-aiml)  
&nbsp;&nbsp;&nbsp;&nbsp;‣ [Week 2: 12-Factor App for Machine Learning Systems](#-week-2-12-factor-app-for-machine-learning-systems)  
&nbsp;&nbsp;&nbsp;&nbsp;‣ [Week 3: Data Wrangling: Pandas and SQL](#-week-3-data-wrangling-pandas-and-sql)  
&nbsp;&nbsp;&nbsp;&nbsp;‣ [Week 4: Data Visualization and Presentation](#-week-4-data-visualization-and-presentation)  
&nbsp;&nbsp;&nbsp;&nbsp;‣ [Week 5: Linear Models](#-week-5-linear-models)  
‣ [📢 LinkedIn Weekly Posts](#-linkedin-weekly-posts)

---

## 📘 Detailed Weekly Logs

### 🔹 Week 1: Introduction to AI/ML

#### 📌 Overview
‣ Introduced AI history, goals, and fellowship expectations  
‣ Outlined the 6-month roadmap and learning structure

#### 🧰 Pre-session
‣ **Intermediate Python:** Lists, Tuples, Dictionaries, Sets, Strings, Collections, Itertools, Lambda, Exceptions, Logging, JSON, Decorators, Generators, Threading & Multiprocessing  
‣ **Mathematics:** Linear Algebra, Matrix Calculus, Probability Theory  
‣ **AI/ML:** Applications, typical ML workflow

#### 🧠 Live Session : 2 Hrs on Sunday
‣ Discussed AI evolution, real-world impact, and fellowship roadmap

#### 📝 Post-session
‣ Created a personalized learning plan and defined focus areas

#### 💡 Key Insight
‣ Setting clear intentions early helps stay focused and track meaningful progress

---

### 🔹 Week 2: 12-Factor App for Machine Learning Systems

#### 📌 Overview
‣ Explored the 12-Factor App principles for building scalable, cloud-native ML systems  
‣ Reviewed Python’s core ML libraries and project structuring with Git & Cookiecutter

#### 🧰 Pre-session
‣ Covered: Git basics, project templates, core Python ML libraries  
‣ Topics:  
  ‣ 12-Factor App methodology  
  ‣ REST APIs with FastAPI  
  ‣ Async programming  
  ‣ Logging & debugging  
  ‣ Docker containerization

#### 🧠 Live Session
‣ Applied 12-Factor principles to real ML system design  
‣ Explored deployment-ready architecture and best practices

#### 📝 Post-session
‣ **Task:** Create a FastAPI microservice implementing as many 12-Factor principles as possible, focusing on clarity, practicality, and best development practices

🔗 [Post-session Repo](https://github.com/KushalRegmi61/Explore-Cafe-API)

#### 💡 Key Insight
‣ Engineering discipline (like 12-Factor & Docker) turns ML code into scalable, reliable systems

---

### 🔹 Week 3: Data Wrangling: Pandas and SQL

#### 📌 Overview
‣ Master data wrangling using Pandas and SQL to clean, transform, and analyze diverse datasets.  
‣ Combine both tools for advanced, real-world data tasks.

#### 🧰 Pre-session
‣ Data types, Pandas basics, SQL queries (filtering, joins, aggregation)  
‣ Data validation with Pydantic and SQL wrangling

#### 🧠 Live Session
‣ Hands-on exercises integrating Pandas and SQL for real-world data challenges

#### 📝 Post-session Tasks  
‣ **SQL Assignment:** Applied SQL concepts to 20 real-world queries on customers, employees, reports, subqueries, rankings, and data updates—reinforcing SQL’s critical role in AI/ML analysis.  
‣ **Pandas Assignment:** Practiced data wrangling with data creation/loading, indexing, filtering, cleaning, transforming, aggregating, merging, and exploratory analysis, uncovering key product rating insights.

🔗 [Post-session Repo](https://github.com/KushalRegmi61/Data_Wrangling_with_SQL_and_Pandas/tree/master)

#### 💡 Key Insight  
‣ Leveraging Pandas and SQL together efficiently solves complex data problems essential for AI/ML workflows.

---

### 🔹 Week 4: Data Visualization and Presentation

#### 📌 Overview  
‣ Understand two key purposes of data visualization: exploratory analysis (discovering insights) and explanatory analysis (communicating findings).  
‣ Learn chart selection, ethical data presentation, and storytelling principles.  

#### 🧰 Pre-session  
‣ Study visualization types, design principles (color theory, typography, layout), and ethical considerations in data presentation.  
‣ Explore different visualization libraries such as Altair, Matplotlib, Plotly, Seaborn

#### 🧠 Live Session  
‣ Practice analyzing and visualizing 1D, 2D, and multi-dimensional data.  
‣ Apply exploratory vs explanatory visualizations and build compelling data stories.

#### 📝 Post-session Tasks  
‣ Applied learned concepts to analyze and visualize Seaborn’s Tips dataset, creating effective exploratory and explanatory visualizations.  

🔗 [Post-session Repo](https://github.com/KushalRegmi61/data_visualization/tree/master)

#### 💡 Key Insight  
‣ Effective visualization and storytelling transform data insights into impactful decisions while maintaining ethical clarity.

---

### 🔹 Week 5: Linear Models

#### 📌 Overview  
This week focused on **linear models** as foundational tools in both **predictive modeling** and **statistical inference**. We covered a wide range of techniques—from basic linear regression to more advanced generalized linear models (GLMs), emphasizing their mathematical intuition, implementation, and practical applications.

#### 🧰 Pre-session  
Explored essential concepts and mathematical foundations behind linear models:

‣ **Linear & Polynomial Regression**: Introduction to modeling linear relationships  
‣ **Performance Metrics**: R², RMSE, and error analysis  
‣ **MLE & Least Squares**: Derivations and geometric intuition of OLS in simple and multiple regression  
‣ **Regularization Techniques**: Lasso, Ridge, and ElasticNet with geometric interpretation  
‣ **Classification via Logistic Regression**: Binary, One-vs-One, One-vs-Rest, and Multinomial Logistic Regression  
‣ **Optimization**: Parameter tuning using Gradient Descent for simple and multiple models 

#### 🧠 Live Session  
‣ Implemented real-world use cases of linear models through hands-on coding  
‣ Interpreted coefficients and assessed statistical significance of model terms  
‣ Compared regularized (Lasso/Ridge) vs. non-regularized models for robustness and overfitting control  

#### 📝 Post-session  
Two applied tasks were conducted using a student performance dataset:

‣ **Task 1: Regression Modeling**  
Trained a linear regression model to predict final grades (G3), using evaluation metrics like R² and RMSE  

‣ **Task 2: Classification Task**  
Applied logistic regression to classify student pass/fail outcomes  

Key steps in both tasks included:  
  ‣ Feature selection via correlation heatmaps  
  ‣ Comparing performance with all vs. selected features  
  ‣ Encoding categorical variables  
  ‣ Evaluation using accuracy, precision, recall, and F1-score  
  ‣ Benchmarking against baseline (dummy) classifiers  

🔗 [Post-session_Repo](https://github.com/KushalRegmi61/Linear_Models/tree/master)

#### 💡 Key Insight  
Linear models, despite their simplicity, remain powerful and interpretable. With extensions like **regularization** and **GLMs**, they form a flexible toolkit for solving a wide range of regression and classification problems across domains.

---

## 📢 LinkedIn Weekly Posts

‣ [Week 1–4 Recap](https://www.linkedin.com/posts/kushal-regmi-0b88a42aa_aifellowshipfusemachinesreadmemd-at-master-activity-7339869109536387073-0Zrf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEqGxYwBvISQU0D0hQ4gElKpiVYsO41o6NM)  
‣ Week 5 – Coming Soon  
‣ Week 6 – Coming Soon  
‣ Week 7 – Coming Soon  
‣ Week 8 – Coming Soon  
‣ Week 9 – Coming Soon  
‣ Week 10 – Coming Soon  
‣ Week 11 – Coming Soon  
‣ Week 12 – Coming Soon  
‣ Week 13 – Coming Soon  
‣ Week 14 – Coming Soon  
‣ Week 15 – Coming Soon  
‣ Week 16 – Coming Soon  
‣ Week 17 – Coming Soon  
‣ Week 18 – Coming Soon  
‣ Week 19 – Coming Soon  
‣ Week 20 – Coming Soon  
‣ Week 21 – Coming Soon  
‣ Week 22 – Coming Soon  
‣ Week 23 – Coming Soon  
‣ Week 24 – Coming Soon  
‣ Week 25 – Coming Soon
