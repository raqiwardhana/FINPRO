# Background and Objectives

XYZ Company faces a high annual employee attrition rate of 16.1%. This effect negatively impacting project workflows, increasing recruitment costs, decreasing training efficiency, raising team workloads, and leading to a loss of key knowledge workers.

## Why This Matters

Employee attrition is a universal business challenge that directly affects productivity. The ideal attrition rate varies by industry, the general rate is around 10%. With a current gap of 6.1% from this ideal, a timely analysis is crucial to identify root causes and implement strategic solutions.

# Problem & Objective

## Problem
- 16.1% attrition per year from total employee in 2015
- normally attrition rate ranges between 5 to 10% per year
- estimated attrition cost around INR 263 Million Per Year

## Goals
- Reduce Attrition to 10% in 1 year
- Early Warning
- Workforce Planning
- Knowledge Management

## Objective
- Identify factors influencing employee attrition
- Build predictive models to detect employees at risk of leaving
- Determine the most influential factors as business intervention priorities
- Provide data-driven retention strategy recommendations

## Benefit
- Recruitment and training cost efficiency
- Increased operational stability

# Risk & Feasibility

## Risk

### Data Risk
- Irrelevant data
- Missing values
- Data imbalance

#### Mitigation
- Evaluation & adjustment
- Data cleaning & Processing
- Handling imbalance

### Model Risk
- Over/Underfitting
- Low interpretability

#### Mitigation
- Cross validation & Hyperparameter Tuning
- Use models that are easy to interpret

### Business Risk
- Result misinterpretation
- Stakeholder resistance
- Unactionable insight

#### Mitigation
- Clear visualization
- Early stakeholder involvement
- Focus on actionable insights

### Technical Risk
- Tools limitation
- Deployment failure
- Integration with HR systems

#### Mitigation
- Use simple tools
- Create a prototype
- Create a clear documentation

## Feasibility

### Technical Feasibility
- Available HR datasets.
- ML algorithms such as Logistic Regression and Random Forest are easy to use.
- Open-source tools that can be used and mastered by data scientist teams.

### Operational Feasibility
- Stakeholders can use the model results for retention strategies.
- Dashboards can help monitor attrition risk.

### Economic Feasibility
- Low costs due to the use of open-source tools.
- High benefits including reduced recruitment and training costs, and increased employee productivity.

# Data
- [Source: Internal company data ](https://drive.google.com/drive/folders/1K4wW1s68h1yaqAihkGgz9uyMbrYJ8OUN?usp=drive_link)
- Scope: 4,410 employees
- Period: 2015 snapshot
- Format: Tabular HRIS export (.csv)
- Target Variable: Attrition (Yes/No)

## Dataset
- data_dictionary
- general_data
- manager_survey_data
- employee_survey_data
- in_time
- out_time

<img alt="Dataset ERD" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/Dataset_Erd.png/">

## Available Features

1. **Demographic & Personal**: Age, Gender, Marital Status, Education, Education Field, Distance from Home, Over18, NumCompaniesWorked
    
2. **Career Tenure & History**: TotalWorkingYears, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, TrainingTimesLastYear
    
3. **Satisfaction Metrics**: JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement
    
4. **Job & Role Information**: Department, JobRole, JobLevel, BusinessTravel, EmployeeCount, EmployeeNumber, StandardHours
    
5. **Compensation & Benefits**: MonthlyIncome, PercentSalaryHike, PerformanceRating, StockOptionLevel
    
6. **Attendance**: in_time, out_time

# EDA
- Some data have positive skew distribution
<img alt="violin plot of EDA" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/EDA_Violinplot.png" />

- On the Boxplot some data shown have outliers that need to be handled
<img alt="violin plot of EDA" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/EDA_Boxplot.png" />

- Many of those who do attrition are:

    - Travel frequently
    - Human Resource as Education field
    - Human Resource Department
    - Job role as Research Director
    - Single
    - Overwork
    - Male
<img alt="Percentage of retention with bivariate" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/EDA_percentage.png" />

- missing value:


|Column|Percentage|
| :--- | :--- |
|JobSatisfaction|(0.4%)|
|EnvironmentSatisfaction|(0.5%)|
|WorkLifeBalance |(0.8%)|
|NumCompaniesWorked |(0.4%)|
|TotalWorkingYears |(0.2%)|

- As per checking there is no duplicate data
identified in dataframes


# Preprocessing
- Outlier handled by using Log transformation ('MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'overwork_days')
- missing value is handled by filled with the mode

# Feature Selection & Engineering
- drop data for having have constant value, StandardHours, EmployeeCount, Over18
- droping EmployeeID becasue irrelevant
- droping data with low correlation (e.g.'Gender', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike, etc.')
<img alt="Percentage of retention with bivariate" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/Future%20relation%20scale.png" />

# Modeling

## Preparation
- Split Data:

    - Train Data: 60%
    - Test Data: 40%

- Scaling: Standarixation
- Data Imbalance
<img alt="Comparasion of before and after Attrition Distribution using SMOTE" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/Attrition_Distribution_BeforeSMOTE.png"/>

## Baseline Model
- Using Logistis regression without the hyperparameter tuning for get the initial model performance

|      |Test|
| :--- | :--- |
| Recall |0.24| |
| F1-Score |0.35| |
| ROC_AUC |0.79||


- Confusion Matrix

||P+|P-|
| :--- | :--- | :--- | 
|A+|1438|42 |
|A-|216|68|

- P = Prediction
- A = Actual


## Model Exploration

- Before tuning

| Model |Recall|F1-Score|ROC_AUC|
|-------|-----------------|------------------|------------------|
|XGBoost|0.85|0.90|0.95|
|ANN (MLP)|0.85|0.89|0.93|
|Random_Forest|0.81|0.89|0.98|
|LightGBM|0.81|0.87|0.95|
|CatBoost|0.77|0.86|0.95|

- After Tuning

| Model |Recall|F1-Score|ROC_AUC|
|-------|-----------------|------------------|------------------|
|▲CatBoost|![up](https://img.shields.io/badge/0.85-brightgreen)|![up](https://img.shields.io/badge/0.9-brightgreen)|0.95|
|▲LightGBM|![up](https://img.shields.io/badge/0.85-brightgreen)|![up](https://img.shields.io/badge/0.91-brightgreen)|![up](https://img.shields.io/badge/0.96-brightgreen)|
|▼ANN (MLP)|![down](https://img.shields.io/badge/0.84-red)|![down](https://img.shields.io/badge/0.87-red)|![up](https://img.shields.io/badge/0.95-brightgreen)|
|▼XGBoost|![down](https://img.shields.io/badge/0.82-red)|![down](https://img.shields.io/badge/0.88-red)|![up](https://img.shields.io/badge/0.96-brightgreen)|
|▼Random_Forest|0.81|0.89|![down](https://img.shields.io/badge/0.97-red)|


- Confusion Matrix (Before Tuning)

||True_Positive|False_Negative|False_Positive|True_Ngative|
|:---|:---|:---|:---|:---|
|XGBoost|1470|42|10|242|
|ANN_(MLP)|1462|43|18|241
|Random_Forest|1477|52|3|232|
|LightGBM|1467|54|13|230|
|CatBoost|1474|65|6|219|

- Confusion Matrix (After Tuning)

||True_Positive|False_Negative|False_Positive|True_Ngative|
|:---|:---|:---|:---|:---|
|▲CatBoost|![down](https://img.shields.io/badge/1470-red)|![down](https://img.shields.io/badge/42-red)|![up](https://img.shields.io/badge/10-brightgreen)|![up](https://img.shields.io/badge/242-brightgreen)|
|▲LightGBM|![up](https://img.shields.io/badge/1474-brightgreen)|![down](https://img.shields.io/badge/43-red)|![down](https://img.shields.io/badge/6-red)|![down](https://img.shields.io/badge/214-red)|
|▼ANN_(MLP)|![down](https://img.shields.io/badge/1456-red)|![up](https://img.shields.io/badge/45-brightgreen)|![up](https://img.shields.io/badge/24-brightgreen)|![down](https://img.shields.io/badge/239-red)|
|▼XGBoost|![down](https://img.shields.io/badge/1465-red)|![up](https://img.shields.io/badge/51-brightgreen)|![up](https://img.shields.io/badge/15-brightgreen)|![down](https://img.shields.io/badge/233-red)|
|▼Random_Forest|![down](https://img.shields.io/badge/1475-red)|![down](https://img.shields.io/badge/51-red)|![up](https://img.shields.io/badge/5-brightgreen)|![up](https://img.shields.io/badge/233-brightgreen)|




> ![up](https://img.shields.io/badge/Green-brightgreen) = Increasing
  
> ![down](https://img.shields.io/badge/Red-red) = Decreasing

## Feature Importance
<img alt="Feature importance score" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/Feature_Importance.png"/>

- The business insight:

    - Different retention strategies for each age segment
    - High workload might lead to employees burnout
    - Employee with shorter and longer tenure shown different attrition pattern
    - the farther the house are the higher resignation chances
    - Company need to review salary compaetitivesness and fairness

## Final Model Selection
### CatBoost

|Metric|Outcome|Interpretation|
|:---|:---|:---|
|Recall|0.85|ML can catch 85% True Positive and 15% chances of wrong prediction|
|ANN_(MLP)|0.90|The model have a good balance between predicting and minimizing false prediction|
|Random_Forest|1477|ML can predict whose likely or having the higher chance to do attrition|

### SHAP

<img alt="SHAP graph" src="https://github.com/raqiwardhana/FINPRO/blob/main/Asset/SHAP_Graph.png"/>

- The SHAP results doesn't directly indicate data leakage
- Identified features remain reasonable for an employee attrition problem.
- Features such as (a)longer work hours, (b)lower satisfaction, (c)lower income, and (d)shorter tenure are common and meaningful rather than suspicious indicators of leakage. 

# Error Analysis & Business Impact
## Error Analysis

- CatBoost Confusion Matrix

||P+|P-|
| :--- | :--- | :--- | 
|A+|1470|10|
|A-|42|242|

- P = Prediction
- A = Actual


- The CatBoost model achieved strong overall performance with low prediction errors.
- Errors prediction still remain:
    - False Negative (42): employees who are likely to resign but were not detected by the model.
    - False Positive (10): employees predicted as attrition risk but actually stayed.

- Minimizing False Negative is important to reduce the risk of losing valuable employees without early intervention.
- The remaining errors may occur due to overlapping or unseen behavioral patterns in the dataset.

## Business Impact

|Metric|Before|AFter|Improvement|
| :--- | :--- | :--- | :--- |
|Attrition Rate|16.1%|10%|![up](https://img.shields.io/badge/▼37.9%-brightgreen)|
|Attrition Cost|263 Million INR|163.35 Million INR|![up](https://img.shields.io/badge/▼99.65%20Million%20INR-brightgreen)|

- Reduced employee turnover
- More efficient hiring cost
- Lower retraining expenses
- Improved workforce stability

# Recommendation

- Use more updated employee data
- Collect larger and more diverse employee data
-Improve detection of false negative cases
- perform periodic retraining and real-time monitoring

# Installation and Usage

To run this project, you will need the following Python packages. All dependencies are listed in the `requirements.txt` file.

Dependencies:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `joblib`
- `catboost`
- `shap`
- `datetime`
- `XlsxWriter`
- `openpyxl`
- `xlrd`

# Monitoring & Maintenance Strategy
- Model performance monitoring

    - monitored prediction model regularly using evaluation metrics such as:

        - Precision
        - Recall
        - F1-Score

- Data quality monitoring

    Regularly check:

    - No missing values
    - Correct data formats
    - Consistent feature distributions
    - No duplicated records

`Poor data quality may reduce prediction accuracy and lead to unreliable results.`

- Periodic model retraining

    The model should be retrained periodically using newer employee data to maintain relevance and accuracy.

    - Every 3~6 months
    - When significant performance drops are detected
    - When company workforce patterns change significantly


- System maintenance

    The application and supporting infrastructure should be regularly by:
    - Updating dependencies and libraries
    - Fixing bugs and system issues
    - Monitoring application uptime
    - Improving UI/UX when necessary

[click here](https://attrictionbylogdata.streamlit.app) to try the ML

# Challenges & Limitation

- Limited dataset
- Data quality depedency
- Changing employess behaviour
- Risk of error prediction
- Maintenance requirement
