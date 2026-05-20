import streamlit as st
import joblib
import pandas as pd
import numpy as np

def prediction(data):
    model = joblib.load("./my_model.joblib")
    train_columns = joblib.load("./columns.pkl")
    
    df = pd.DataFrame([data])
    df['isMale'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['DistanceFromHome_log'] = np.log1p(df['DistanceFromHome'])
    df['MonthlyIncome_log'] = np.log1p(df['MonthlyIncome'])
    df['NumCompaniesWorked_log'] = np.log1p(df['NumCompaniesWorked'])
    df['PercentSalaryHike_log'] = np.log1p(df['PercentSalaryHike'])
    df['TotalWorkingYears_log'] = np.log1p(df['TotalWorkingYears'])
    df['YearsAtCompany_log'] = np.log1p(df['YearsAtCompany'])
    df['YearsSinceLastPromotion_log'] = np.log1p(df['YearsSinceLastPromotion'])
    df['YearsWithCurrManager_log'] = np.log1p(df['YearsWithCurrManager'])
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=train_columns, fill_value=0)
    hasil = model.predict(df)

    row = df.iloc[[0]]
    proba = model.predict_proba(row)[0]
    label = proba.argmax()
    confidence = round(proba[label] * 100,1)

    if hasil==1:
        col2.markdown(f"""<div style="background: #B42700;
                            height:200px;
                            border-radius:20px;
                            color:white;
                            font-weight:bold;
                            padding-left:20px">
                      <h4>Result</h4>
                      <h1 style="display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                            margin-bottom:-25px">{confidence}%</h1>
                      <p style="display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                            margin-top: 0px;
                            margin-bottom: 2px;
                            line-height:-10;">The employee is predicted to churn.</p>
                      </div>""", unsafe_allow_html=True)
    else:
        col2.markdown(f"""<div style="background: #029237;
                            height:200px;
                            border-radius:20px;
                            color:white;
                            font-weight:bold;
                            padding-left:20px">
                      <h4>Result</h4>
                      <h1 style="display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                            margin-bottom:-25px">{confidence}%</h1>
                      <p style="display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                            margin-top: 0px;
                            margin-bottom: 2px;
                            line-height:0;">The employee is predicted to remain with the company</p>
                      </div>""", unsafe_allow_html=True)
    

st.set_page_config(
    page_title="Attriction",
    page_icon="🎯",
    layout="wide",
)

# with st.sidebar:
#     st.title("Attriction")
#     st.write("Attrition Prediction Application by LogData")
st.write("# Atriction🚀")
st.write("Predict your employee attrition now!")


tab1, tab2, tab3 = st.tabs(["Personal", "Group", "About Us"])

with tab1:
    st.header("Personal Prediction")
    with st.form("personal"):
        col1, col2 = st.columns(2)
        
        col1.markdown("**Biodata🙋**")
        employeeID = col1.text_input("EmployeeID")
        Age = col1.number_input("Age", step=1, min_value=0)
        Gender = col1.radio(label="Gender",
                            options=["Male","Female"],
                            horizontal=True)
        Education = col1.select_slider("Education",
                                       options=[1,2,3,4,5])
        EducationField = col1.selectbox("Education Field",
                                        options=['Human Resources','Life Sciences','Marketing','Medical','Technical Degree','Other'])
        MaritalStatus = col1.radio(label="Marital Status",
                        options=['Divorced', 'Married', 'Single'],
                        horizontal=True)
        DistanceFromHome = col1.number_input('Distance From Home', step=1, min_value=0)
        Department = col1.radio(label="Department",
                                options=["Human Resources", "Research & Development", "Sales"],
                                horizontal=True)
        Joblevel = col1.select_slider("Job Level",
                                      options=[1,2,3,4,5])
        JobRole = col1.selectbox("Job Role",
                                 options=['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'])
        BusinessTravel = col1.radio(label= "BusinessTravel",
                                  options=["Non-Travel","Travel_Rarely", "Travel_Frequently"],
                                  format_func=lambda x: {
                                      "Non-Travel": "Non-Travel",
                                      "Travel_Rarely": "Rarely",
                                      "Travel_Frequently": "Frequently"
                                      }[x],
                                      horizontal=True)
        EmployeeCount = 1
        MonthlyIncome = col1.number_input('Monthly income', step=1, min_value=0)
        PercentSalaryHike = col1.number_input('Percent Salary Hike', step=1, min_value=11, max_value=25)        
        NumCompaniesWorked = col1.number_input('Number of Companies Worked', step=1, min_value=1, max_value=9)
        Over18 = 'Y'
        StandardHours = 8
        StockOptionLevel = col1.select_slider("Stock Option Level",
                                              options=[0,1,2,3])
        TotalWorkingYears = col1.number_input('Total Working Years', step=1, min_value=0)
        TrainingTimesLastYear = col1.select_slider("Training Times Last Year",
                                     options=[0, 1, 2, 3, 4, 5, 6])
        YearsAtCompany = col1.number_input('Years At Company', step=1, min_value=0)
        YearsSinceLastPromotion = col1.number_input('Years Since Last Promotion', step=1, min_value=0) 
        YearsWithCurrManager = col1.number_input('Years With Curr Manager', step=1, min_value=0)

        col2.markdown("**Survey Data📝**")
        EnvironmentSatisfaction = col2.select_slider("Environment Satisfaction",
                                                     options=[1,2,3,4])
        JobSatisfaction = col2.select_slider("Job Satisfaction",
                                             options=[1,2,3,4])
        WorkLifeBalance = col2.select_slider("Work Life Balance",
                                             options=[1,2,3,4])
        
        col2.markdown("**Performance📈**")
        JobInvolvement = col2.select_slider("Job Involvement",
                                            options=[1,2,3,4])
        PerformanceRating = col2.select_slider("Performance Rating",
                                               options=[1,2,3,4])
        
        col2.markdown("**Work Hours🕛**")
        total_work_hours = col2.number_input('Total Work Hours', step=1, min_value=0)

        col2.text("Make sure semua kolom sudah diisi dengan nilai yang sesuai sebelum menekan tombol")
        submitted = col2.form_submit_button("Predict now!", type="primary",width="stretch")
        if submitted:
            data = {
                "Age": Age,
                "BusinessTravel": BusinessTravel,
                "Gender":Gender,
                "Department": Department, 
                "DistanceFromHome": DistanceFromHome,
                "Education": Education, 
                "EducationField": EducationField, 
                "EmployeeCount": EmployeeCount, 
                "EmployeeID": employeeID, 
                "Gender": Gender,
                "JobLevel": Joblevel, 
                "JobRole": JobRole, 
                "MaritalStatus": MaritalStatus, 
                "MonthlyIncome": MonthlyIncome,
                "NumCompaniesWorked": NumCompaniesWorked, 
                "Over18": Over18, 
                "PercentSalaryHike": PercentSalaryHike,
                "StandardHours":StandardHours,
                "StockOptionLevel": StockOptionLevel,
                "TotalWorkingYears": TotalWorkingYears, 
                "TrainingTimesLastYear": TrainingTimesLastYear,
                "YearsAtCompany": YearsAtCompany, 
                "YearsSinceLastPromotion": YearsSinceLastPromotion, 
                "YearsWithCurrManager": YearsWithCurrManager,
                'EnvironmentSatisfaction': EnvironmentSatisfaction, 
                'JobSatisfaction': JobSatisfaction,
                'WorkLifeBalance': WorkLifeBalance,
                'JobInvolvement': JobInvolvement, 
                'PerformanceRating': PerformanceRating,
                'total_work_hours': total_work_hours
            }
            prediction(data)

with tab2:
    st.header("Group Prediction")

with tab3:
    st.header("About Us")

st.markdown("""<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: gray;
}</style>
<div class="footer">
© 2026 Attriction by LogData Team | Built with Streamlit
</div>""", unsafe_allow_html=True)
