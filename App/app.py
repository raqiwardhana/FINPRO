import streamlit as st
import joblib
import pandas as pd
import numpy as np

def visualize(data):
    a, b, c= st.columns(3)

    a.metric("Employees", data['EmployeeID'].count(), border=True)
    b.metric("Attrition Prediction", data['prediction'].value_counts().get("Yes", 0), border=True)
    c.metric("Average Total Work Hours", round(data['total_work_hours'].mean()), border=True)

    # over_aw, age = st.columns(2)

    # over_aw

def prediction(data):
    model = joblib.load("App/my_model.joblib")
    train_columns = joblib.load("App/columns.pkl")
    
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
                            line-height:-10;">The employee is predicted to leave the company.</p>
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

def prediction_group(df_general, df_employee, df_manager, df_in_time, df_out_time):
    model = joblib.load("App/my_model.joblib")
    train_columns = joblib.load("App/columns.pkl")
    
    df_general = pd.read_csv(df_general)
    df_employee = pd.read_csv(df_employee)
    df_manager = pd.read_csv(df_manager)
    df_in_time = pd.read_csv(df_in_time)
    df_out_time = pd.read_csv(df_out_time)
    cols_to_fix = df_in_time.columns[1:]
    df_in_time[cols_to_fix] = df_in_time[cols_to_fix].apply(pd.to_datetime)
    df_out_time[cols_to_fix] = df_out_time[cols_to_fix].apply(pd.to_datetime)
    work_hours = df_out_time.iloc[:, 1:] - df_in_time.iloc[:, 1:]

    work_hours = (df_out_time.iloc[:, 1:] - df_in_time.iloc[:, 1:]) / pd.Timedelta(hours=1)
    total_work_hours = work_hours.sum(axis=1)
    average_work_hours = work_hours.mean(axis=1)

    df_in_time.iloc[:, 1:] = df_in_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')
    df_out_time.iloc[:, 1:] = df_out_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')

    in_long = df_in_time.melt(id_vars=['Unnamed: 0'],
                        var_name='date',
                        value_name='in_time')

    out_long = df_out_time.melt(id_vars=['Unnamed: 0'],
                        var_name='date',
                        value_name='out_time')

    df = in_long.merge(out_long, on=['Unnamed: 0', 'date'])
    df = df.dropna(subset=['in_time', 'out_time'])
    df['work_hours'] = (df['out_time'] - df['in_time']).dt.total_seconds() / 3600
    df = df[df['work_hours'] >= 0]
    df['is_overwork'] = df['work_hours'] >= 9

    overwork_days = df.groupby('Unnamed: 0')['is_overwork'].sum()
    overwork_days = overwork_days.reset_index(drop=True)
    df_in_out = pd.concat(
    [total_work_hours, average_work_hours, overwork_days],
    axis=1)

    df_in_out.columns = ['total_work_hours', 'average_work_hours', 'overwork_days']
    df_in_out = df_in_out.reset_index()

    df_in_out['overwork'] = df_in_out['total_work_hours'].apply(
    lambda x: 'Yes' if x > 2080 else 'No')

    df_in_out = df_in_out.drop(columns=['index'])
    df_in_out.insert(0, 'EmployeeID', range(1, len(df_in_out) + 1))
    final_df = pd.merge(df_general, df_employee, on='EmployeeID')
    final_df = pd.merge(final_df,df_manager, on='EmployeeID')
    final_df = pd.merge(final_df,df_in_out, on='EmployeeID')
    final_df = final_df.fillna(final_df.select_dtypes(include=['object', 'category']).mode().iloc[0])
    df = final_df.copy()
    
    final_df['isMale'] = final_df['Gender'].map({'Male': 1, 'Female': 0})
    final_df['overwork'] = final_df['overwork'].map({'Yes': 1, 'No': 0})

    final_df = pd.get_dummies(final_df, columns=['Department'],dtype=int, drop_first=True)
    final_df = pd.get_dummies(final_df, columns=['EducationField'],dtype=int, drop_first=True)
    final_df = pd.get_dummies(final_df, columns=['JobRole'],dtype=int, drop_first=True)
    final_df = pd.get_dummies(final_df, columns=['MaritalStatus'],dtype=int, drop_first=True)
    final_df = pd.get_dummies(final_df, columns=['BusinessTravel'],dtype=int, drop_first=True)
    pd.set_option('display.max_columns', None)

    final_df['DistanceFromHome_log'] = np.log1p(final_df['DistanceFromHome'])
    final_df['MonthlyIncome_log'] = np.log1p(final_df['MonthlyIncome'])
    final_df['NumCompaniesWorked_log'] = np.log1p(final_df['NumCompaniesWorked'])
    final_df['PercentSalaryHike_log'] = np.log1p(final_df['PercentSalaryHike'])
    final_df['TotalWorkingYears_log'] = np.log1p(final_df['TotalWorkingYears'])
    final_df['YearsAtCompany_log'] = np.log1p(final_df['YearsAtCompany'])
    final_df['YearsSinceLastPromotion_log'] = np.log1p(final_df['YearsSinceLastPromotion'])
    final_df['YearsWithCurrManager_log'] = np.log1p(final_df['YearsWithCurrManager'])
    final_df = final_df.reindex(columns=train_columns, fill_value=0)
    y_pred = model.predict(final_df)
    df['prediction'] = y_pred
    df['prediction'] = df['prediction'].map({1: 'Yes', 0: 'No'})
    pred_series = pd.Series(y_pred)
    persentase = pred_series.value_counts(normalize=True) * 100
    persentase = round(persentase[1],2)
    if (persentase>=10) :
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
                            margin-bottom:-25px">{persentase}%</h1>
                      <p style="display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                            margin-top: 0px;
                            margin-bottom: 2px;
                            line-height:-10;">Employees are predicted to leave the company.</p>
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
                            margin-bottom:-25px">{persentase}%</h1>
                      <p style="display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                            margin-top: 0px;
                            margin-bottom: 2px;
                            line-height:0;">Employees are predicted to remain with the company</p>
                      </div>""", unsafe_allow_html=True)
    visualize(df)
    col2.write("  ")
    col2.write(df[['EmployeeID','total_work_hours','Age','YearsAtCompany','MaritalStatus','BusinessTravel','prediction']])
    
    @st.cache_data
    def convert_for_download(df):
        return df.to_csv().encode("utf-8")
    
    csv = convert_for_download(df)

    col2.download_button(
        label="Download Result",
        data=csv,
        file_name="ResultData.csv",
        mime="text/csv",
        icon=":material/download:",
    )

st.set_page_config(
    page_title="Attriction",
    page_icon="🎯",
    layout="wide",
)

st.write("# Attriction🚀")
st.write("Predict your employee attrition now!")


tab1, tab2, tab3 = st.tabs(["Personal", "Group", "About Us"])

with tab1:
    st.header("Personal Prediction")
    with st.form("personal"):
        col1, col2 = st.columns(2)
        
        col1.markdown("**Biodata🙋**")
        Age = col1.number_input(
            "Age",
            step=1,
            min_value=0,
            help="Input employee age in years")
        Gender = col1.radio(label="Gender",
                            options=["Male","Female"],
                            horizontal=True)
        Education = col1.select_slider("Education Level",
                                       options=[1,2,3,4,5],
                                       help="""
                                        - 1 = Below College
                                        - 2 = College
                                        - 3 = Bachelor
                                        - 4 = Master
                                        - 5 = Doctor
                                        """
                                        )
        EducationField = col1.selectbox("Education Field",
                                        options=['Human Resources','Life Sciences','Marketing','Medical','Technical Degree','Other'])
        MaritalStatus = col1.radio(label="Marital Status",
                        options=['Divorced', 'Married', 'Single'],
                        horizontal=True)
        DistanceFromHome = col1.number_input('Distance From Home',
                                                step=1,
                                                min_value=0,
                                                help="Distance in KM")
        Department = col1.radio(label="Department",
                                options=["Human Resources", "Research & Development", "Sales"],
                                horizontal=True)
        Joblevel = col1.select_slider("Job Level",
                                      options=[1,2,3,4,5],
                                      help="Job level at company on a scale of 1 to 5")
        JobRole = col1.selectbox("Job Role",
                                 options=['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'])
        BusinessTravel = col1.radio(label= "BusinessTravel",
                                  options=["Non-Travel","Travel_Rarely", "Travel_Frequently"],
                                  help="How frequently the employees travelled for business purposes in the last year",
                                  format_func=lambda x: {
                                      "Non-Travel": "Non-Travel",
                                      "Travel_Rarely": "Rarely",
                                      "Travel_Frequently": "Frequently"
                                      }[x],
                                      horizontal=True)
        EmployeeCount = 1
        MonthlyIncome = col1.number_input('Monthly income (INR)',step=1, min_value=0)
        PercentSalaryHike = col1.number_input('Percent Salary Hike',
                                                step=1, 
                                                min_value=11, 
                                                max_value=25,
                                                help="Percent salary hike for last year, min value is 11, and max value is 25")        
        NumCompaniesWorked = col1.number_input('Number of Companies Worked', 
                                                step=1, 
                                                min_value=1, 
                                                max_value=9,
                                                help="Total number of companies the employee has worked for")
        Over18 = 'Y'
        StandardHours = 8
        StockOptionLevel = col1.select_slider("Stock Option Level",
                                                options=[0,1,2,3],
                                                help="Stock option level of the employee")
        TotalWorkingYears = col1.number_input('Total Working Years', 
                                                step=1, 
                                                min_value=0,
                                                help="Total number of years the employee has worked so far")
        TrainingTimesLastYear = col1.select_slider("Training Times Last Year",
                                                    options=[0, 1, 2, 3, 4, 5, 6],
                                                    help="Number of times training was conducted for this employee last year")
        YearsAtCompany = col1.number_input('Years At Company', 
                                            step=1, 
                                            min_value=0,
                                            help="Total number of years spent at the company by the employee")
        YearsSinceLastPromotion = col1.number_input('Years Since Last Promotion', 
                                                        step=1, 
                                                        min_value=0,
                                                        help="Number of years since last promotion") 
        YearsWithCurrManager = col1.number_input('Years With Curr Manager', 
                                                    step=1, 
                                                    min_value=0,
                                                    help="Number of years since last promotion")

        col2.markdown("**Survey Data📝**")
        EnvironmentSatisfaction = col2.select_slider("Environment Satisfaction",
                                                     options=[1,2,3,4],
                                                     help="""
                                                     Employee Satisfaction survey for the environment\n
                                                     - 1 Low
                                                     - 2 Medium
                                                     - 3 High
                                                     - 4 Very High""")
        JobSatisfaction = col2.select_slider("Job Satisfaction",
                                                options=[1,2,3,4],
                                                help="""
                                                Employee Satisfaction survey for the Job\n
                                                - 1 Low
                                                - 2 Medium
                                                - 3 High
                                                - 4 Very High
                                                """)
        WorkLifeBalance = col2.select_slider("Work Life Balance",
                                                options=[1,2,3,4],
                                                help="""
                                                Employee Satisfaction survey for their Work Life Balance\n
                                                - 1 Low
                                                - 2 Medium
                                                - 3 High
                                                - 4 Very High
                                                """)
        
        col2.markdown("**Performance📈**")
        JobInvolvement = col2.select_slider("Job Involvement",
                                            options=[1,2,3,4],
                                            help="""
                                            Job Involvement Level\n
                                            - 1 Low
                                            - 2 Medium
                                            - 3 High
                                            - 4 Very High
                                            """)
        PerformanceRating = col2.select_slider("Performance Rating",
                                                options=[1,2,3,4],
                                                help="""
                                                Performance Rating for the Employee\n
                                                - 1 Low
                                                - 2 Good
                                                - 3 Excellent
                                                - 4 Outstanding
                                                """)
        
        col2.markdown("**Work Hours🕛**")
        total_work_hours = col2.number_input('Total Work Hours', step=1, min_value=0)

        col2.text("Please make sure all fields are filled in with appropriate values before clicking the button")
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
            with col2.spinner("Calculating Process. Please wait."):
                prediction(data)

with tab2:
    st.header("Group Prediction")
    col1, col2 = st.columns(2)
    col1.markdown("**Upload Data(.csv)📤**")
    df_general = col1.file_uploader("General Data", accept_multiple_files=False, type="csv")
    df_employee = col1.file_uploader("Survey Data", accept_multiple_files=False, type="csv")
    df_manager = col1.file_uploader("Performance Data", accept_multiple_files=False, type="csv")
    df_in_time = col1.file_uploader("In Time Data", accept_multiple_files=False, type="csv")    
    df_out_time = col1.file_uploader("Out TIme Data", accept_multiple_files=False, type="csv")

    if col1.button("Predict now!", type="primary"):
        with col2.spinner("Calculating Process. Please wait."):
            prediction_group(df_general, df_employee, df_manager, df_in_time, df_out_time)

with tab3:
    st.header("Log Data")
    st.image("App/logo.png", width=200)
    st.markdown("""
    At **Log Data**, we believe that data is more than just numbers,  
    it is the key to smarter decisions, better strategies, and meaningful business growth.

    We are a passionate team of data scientists, analysts, and technology enthusiasts  
    dedicated to transforming raw data into valuable insights through machine learning,  
    analytics, and intelligent solutions.

    Our mission is to help businesses make data-driven decisions by providing innovative analytical tools,  
    predictive models, and interactive applications that simplify complex information into actionable insights.

    """)

    st.divider()
    st.subheader("What We Do")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        -  Data Analysis & Visualization  
        -  Machine Learning & Predictive Modeling  
        -  Business Intelligence Solutions  
        """)

    with col2:
        st.markdown("""
        -  AI-Powered Applications  
        -  Data-Driven Decision Support  
        -  Dashboard & Reporting Development  
        """)

    st.divider()

    st.subheader("Our Vision")

    st.info("""
    To become a trusted data science partner that empowers organizations  
    through innovative, accurate, and impactful data solutions.
    """)

    st.divider()

    st.subheader("Our Mission")

    missions = [
        "Deliver reliable and scalable data solutions",
        "Transform complex datasets into meaningful insights",
        "Support smarter and faster business decisions",
        "Continuously innovate using modern AI and analytics technologies"
    ]

    for mission in missions:
        st.markdown(f"✅ {mission}")

    st.divider()
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
© 2026 LogData | Built with Streamlit
</div>""", unsafe_allow_html=True)
