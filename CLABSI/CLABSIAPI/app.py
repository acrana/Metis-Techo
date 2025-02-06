import shap
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
# 输出混淆矩阵
# 预测

if 'model1' not in st.session_state:
    model1 = joblib.load('clf1.pkl')
    model2 = joblib.load('clf2.pkl')
    st.session_state["model1"] = model1
    st.session_state["model2"] = model2
else:
    model1 = st.session_state["model1"]
    model2 = st.session_state["model2"]
scaler1 = joblib.load("Scaler1.pkl")
scaler2 = joblib.load("Scaler2.pkl")
continuous_vars1 = ['LOS_before_using_IMV','LOS_before_using_CVC','APSIII','Temperature','LOS_before_using_IUC','MAP','PT']
continuous_vars2 = ['Age','Aniongap','APSIII','SAPII']

st.set_page_config(layout='wide')

st.write("<h1 style='text-align: center'>Machine learning for individualized prediction of device-associated infection and 30-day survival outcomes after invasive device procedure in intensive care unit patients</h1>",
         unsafe_allow_html=True)
st.warning('The ML model assists ICU physicians in making informed decisions on optimal device insertion timing to reduce the risk of device-associated infections. This assistance includes: (1) For patients newly admitted to the ICU without prior device usage, physicians can input varying durations of CVC and IMV usage based on the patient’s current fixed variables. This predicts the infection risk at different intervals, facilitating the determination of the latest suitable time for device implementation and the appropriate equipment category. (2) For patients with existing ICU tenure and prior device usage, physicians can input different device usage durations and adjust tracheostomy status to forecast infection risks. This helps in assessing the necessity of additional equipment and tracheostomy. (3) In scenarios where device usage duration is fixed, integrating forecasts of infection risk and 30-day mortality risk enables comprehensive evaluation of the patient’s prognosis, allowing timely intervention.')


st.markdown('-----')
# dic1 = {
#     'Male': 1,
#     'Female': 0
# }
dic2 = {
    'Yes': 1,
    'No': 0
}
with st.sidebar:
    st.markdown("# Input variable")
    st.markdown('-----')
    LOS_before_using_CVC = st.text_input("Length of stay in the ICU (hours) before using the invasive mechanical ventilation (IMV)")
    LOS_before_using_IMV = st.text_input("Length of stay in the ICU (hours) before using the central venous catheter (CVC)")
    Tracheostomy = st.selectbox("Tracheostomy", ["Yes", "No"])
    APSIII = st.text_input("APSIII within 24 hours of ICU admission")
    MICU_or_SICU = st.selectbox("Medical intensive care unit or surgical intensive care unit", ["Yes", "No"])
    Temperature = st.text_input("Temperature (℃) within 24 hours of ICU admission")
    LOS_before_using_IUC = st.text_input("Length of stay in the ICU (hours) before using the indwelling urinary catheter (IUC)")
    MAP = st.text_input("Mean arterial pressure (mmHg) within 24 hours of ICU admission")
    RRT = st.selectbox("Renal replacement therapy", ["Yes", "No"])
    PT = st.text_input("Partial thromboplastin time (s) within 24 hours of ICU admission") 
    Cancer = st.selectbox("Cancer", ["Yes", "No"])
    Age = st.text_input("Age (years)")	
    SAPII = st.text_input("SPASII within 24 hours of ICU admission")	
    Cerebrovascular_disease = st.selectbox("Cerebrovascular disease", ["Yes", "No"])	
    Liver_disease = st.selectbox("Liver disease", ["Yes", "No"])	
    Aniongap = st.text_input("Anion gap (mmol/L) within 24 hours of ICU admission") 	
    Myocardial_infarct = st.selectbox("Myocardial infarct", ["Yes", "No"])		
    Two_or_more_devices = st.selectbox("Using two or more devices", ["Yes", "No"])	
if st.sidebar.button("Predict"):
    with st.spinner("Forecast, please wait..."):
        st.header('Prediction for device-associated infections')
        test_df = pd.DataFrame([float(LOS_before_using_CVC), float(LOS_before_using_IMV), dic2[Tracheostomy], float(APSIII), dic2[MICU_or_SICU], 
                                float(Temperature), float(LOS_before_using_IUC), float(MAP), dic2[RRT], float(PT)], 
                               index=['LOS_before_using_CVC', 'LOS_before_using_IMV', 'Tracheostomy', 'APSIII', 'MICU_or_SICU', 'Temperature',
                                      'LOS_before_using_IUC', 'MAP', 'RRT', 'PT']).T
        test_df[continuous_vars1] = scaler1.transform(test_df[continuous_vars1])
        explainer = shap.Explainer(model1)  # 创建解释器shap_ = explainer.shap_values(test_df)
        shap_ = explainer.shap_values(test_df)
        shap.waterfall_plot(
            shap.Explanation(values=shap_[0, :], base_values=explainer.expected_value, data=test_df.iloc[0, :]),
            show=False)
        plt.tight_layout()
        plt.savefig('shap1.png', dpi=300)
        col1, col2, col3 = st.columns([2, 5, 3])
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_[0, :], test_df.iloc[0, :], show=False, matplotlib=True,
                        figsize=(20, 5))
        plt.xticks(fontproperties='Times New Roman', size=16)
        plt.yticks(fontproperties='Times New Roman', size=16)
        plt.tight_layout()
        plt.savefig('shap2.png', dpi=300)
        with col2:
            st.image('shap1.png')
            st.image('shap2.png')
        st.success("Probability of device-associated infection: {:.3f}%".format(model1.predict_proba(test_df)[:, 1][0] * 100))
        
        # st.header('confusion matrix diagram')
        # col7, col8, col9 = st.columns([2, 5, 3])
        # with col8:
        #     st.image('motplot2.jpg')
        # st.warning('Sensitivity: 31.21%, Specificity: 99.98%.')
        
        
        								
        
        st.header('30-day Kaplan-Meier survival curve')
        
        test_df2 = pd.DataFrame([dic2[MICU_or_SICU], dic2[Cancer], float(APSIII), float(Age), float(SAPII), dic2[Cerebrovascular_disease],
                                 dic2[Liver_disease], float(Aniongap), dic2[Myocardial_infarct], dic2[Two_or_more_devices]], 
                                index=['MICU_or_SICU', 'Cancer', 'APSIII', 'Age', 'SAPII', 'Cerebrovascular_disease', 'Liver_disease', 'Aniongap', 'Myocardial_infarct', 'Two_or_more_devices']).T
        
        test_df2[continuous_vars2] = scaler2.transform(test_df2[continuous_vars2])
        surv_funcs = model2.predict_survival_function(test_df2)
        col4, col5, col6 = st.columns([2, 5, 3])
        fig, ax = plt.subplots()
        for fn in surv_funcs:
            ax.step(fn.x, fn(fn.x), where="post", color="#8dd3c7", lw=2)
        plt.ylabel("Probability of survival (%)")
        plt.xlabel("Time since received the first invasive procedure (days)")
        plt.grid()
        with col5:
            st.pyplot(fig)

    
    if not top_decreasing.empty:
        st.write("Factors decreasing risk:")
        for _, row in top_decreasing.iterrows():
            st.write(f"• {row['Feature']}: {row['Impact']*100:.1f}% risk")
