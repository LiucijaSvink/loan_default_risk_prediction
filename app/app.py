# -*- coding: utf-8 -*-
"""
@author: Liucija Svinkunaite
"""
import numpy as np
import pandas as pd
from sklearn import set_config
set_config(transform_output = "pandas")

import joblib
import streamlit as st
import pickle as pkl

from catboost import CatBoostClassifier
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures

# Loading up the classification pipeline created
default_model = joblib.load('credit_model_catboost.pkl')
preprocess = joblib.load('preprocess_features.pkl')
threshold = 0.5

with open("full_home_credit_feature_lists.pkl", "rb") as f:
    full_feature_list = pkl.load(f) 

# Load one data column names
df = pd.DataFrame(columns=full_feature_list, index=range(1))

# Caching the model for faster loading
@st.cache

# Define prediction function
def make_prediction(model, data, threshold):
    predicted_prob = model.predict_proba(data)[:, 1]
    prediction = (predicted_prob >= threshold).astype(int)
    if prediction == 1:
        rate = 4 + predicted_prob
        prediction = f'Loan default is likely. Recommended interest rate {rate}%'
        
    elif prediction == 0:
        prediction = f'No loan default predicted. Recommended interest rate 4%'

    return prediction

# Define data preparation function
def prepare_data(ext_source_cols_mean,
               amt_credit_amt_annuity_ratio, days_birth,
               ext_source_3, amt_annuity,
               days_credit_enddate_first, 
               ext_source_2, days_employed, ext_source_1,
               days_id_publish, amt_credit_sum_sum,
               days_registration, amt_credit_sum_debt_sum,
               amt_credit, days_enddate_fact_max,
               days_credit_update_min,
               cnt_instalment_future_pos_first2_sum_pos_first2,
               amt_credit_sum_max,
               region_population_relative,
               own_car_age,
               days_instalment_max, amt_income_total,
               organization_type,
               name_education_type, code_gender,
               days_last_phone_change,
               day_payment_difference_min,
               amt_credit_max_overdue_max,
               amt_instalment_first, day_payment_difference_pay_first5_sum_pay_first5,
               day_payment_difference_first,
               amt_annuity_prev_app_first2_mean_prev_app_first2,
               amt_instalment_sum, year_balance_first_first,
               days_last_due_prev_app_last5_max_prev_app_last5,
               days_first_due_prev_app_first5_sum_prev_app_first5,
               day_payment_difference_pay_first2_sum_pay_first2,
               day_payment_difference_pay_last5_sum_pay_last5,
               apartments_avg,
               amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5,
               days_instalment_min,
               day_payment_difference_max,
               days_last_due_1st_version_prev_app_first5_sum_prev_app_first5,
               amt_instalment_pay_last5_max_pay_last5,
               days_instalment_mean,
               amt_credit_sum_limit_max,
               num_instalment_version_sum, 
               basementarea_avg,
               days_first_due_prev_app_first5_max_prev_app_first5,
               cnt_instalment_pos_first2_sum_pos_first2,
               status_mean_mean,
               years_beginexpluatation_avg,
               day_payment_difference_mean,
               amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5,
               num_instalment_number_max,
               amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2,
               days_last_due_prev_app_first5_max_prev_app_first5,
               hour_appr_process_start,
               sellerplace_area_prev_app_first5_max_prev_app_first5,
               days_last_due_prev_app_last5_mean_prev_app_last5,
               days_instalment_sum,
               amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5,
               payment_delay_mean, 
               payment_difference_mean,
               cnt_payment_prev_app_last5_sum_prev_app_last5,
               day_payment_difference_sum,
               amt_instalment_max,
               num_instalment_number_mean,
               days_last_due_1st_version_prev_app_last5_sum_prev_app_last5,
               name_family_status):
    
    # Plug in the data to the empty example
    df['ext_source_cols_mean'] = ext_source_cols_mean
    df['amt_credit_amt_annuity_ratio'] = amt_credit_amt_annuity_ratio
    df['days_birth'] = days_birth
    df['ext_source_3'] = ext_source_3
    df['amt_annuity'] = amt_annuity
    df['days_credit_enddate_first'] = days_credit_enddate_first
    df['ext_source_2'] = ext_source_2 
    df['days_employed'] = days_employed
    df['ext_source_1'] = ext_source_1
    df['days_id_publish'] = days_id_publish
    df['amt_credit_sum_sum'] = amt_credit_sum_sum
    df['days_registration'] = days_registration
    df['amt_credit_sum_debt_sum'] = amt_credit_sum_debt_sum
    df['amt_credit'] = amt_credit
    df['days_enddate_fact_max'] = days_enddate_fact_max
    df['days_credit_update_min'] = days_credit_update_min
    df['cnt_instalment_future_pos_first2_sum_pos_first2'] = cnt_instalment_future_pos_first2_sum_pos_first2
    df['amt_credit_sum_max'] = amt_credit_sum_max
    df['region_population_relative'] = region_population_relative
    df['own_car_age'] = own_car_age
    df['days_instalment_max'] = days_instalment_max
    df['amt_income_total'] = amt_income_total
    df['organization_type'] = organization_type
    df['name_education_type'] = name_education_type
    df['code_gender'] = code_gender
    df['days_last_phone_change'] = days_last_phone_change
    df['day_payment_difference_min'] = day_payment_difference_min
    df['amt_credit_max_overdue_max'] = amt_credit_max_overdue_max
    df['amt_instalment_first'] =  amt_instalment_first
    df['day_payment_difference_pay_first5_sum_pay_first5'] = day_payment_difference_pay_first5_sum_pay_first5
    df['day_payment_difference_first'] = day_payment_difference_first
    df['amt_annuity_prev_app_first2_mean_prev_app_first2'] = amt_annuity_prev_app_first2_mean_prev_app_first2
    df['amt_instalment_sum'] = amt_instalment_sum
    df['year_balance_first_first'] = year_balance_first_first
    df['days_last_due_prev_app_last5_max_prev_app_last5'] \
        = days_last_due_prev_app_last5_max_prev_app_last5
    df['days_first_due_prev_app_first5_sum_prev_app_first5'] \
        = days_first_due_prev_app_first5_sum_prev_app_first5
    df['day_payment_difference_pay_first2_sum_pay_first2'] \
        = day_payment_difference_pay_first2_sum_pay_first2
    df['day_payment_difference_pay_last5_sum_pay_last5'] \
        = day_payment_difference_pay_last5_sum_pay_last5
    df['apartments_avg'] = apartments_avg
    df['amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5'] \
        = amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5
    df['days_instalment_min'] = days_instalment_min
    df['day_payment_difference_max'] = day_payment_difference_max
    df['days_last_due_1st_version_prev_app_first5_sum_prev_app_first5'] = \
        days_last_due_1st_version_prev_app_first5_sum_prev_app_first5
    df['amt_instalment_pay_last5_max_pay_last5'] \
        = amt_instalment_pay_last5_max_pay_last5
    df['days_instalment_mean'] = days_instalment_mean
    df['amt_credit_sum_limit_max'] = amt_credit_sum_limit_max
    df['num_instalment_version_sum'] = num_instalment_version_sum
    df['basementarea_avg'] = basementarea_avg
    df['days_first_due_prev_app_first5_max_prev_app_first5'] \
        = days_first_due_prev_app_first5_max_prev_app_first5
    df['cnt_instalment_pos_first2_sum_pos_first2'] \
        = cnt_instalment_pos_first2_sum_pos_first2
    df['status_mean_mean'] = status_mean_mean
    df['years_beginexpluatation_avg'] = years_beginexpluatation_avg
    df['day_payment_difference_mean'] = day_payment_difference_mean
    df['amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5'] \
        = amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5
    df['num_instalment_number_max'] = num_instalment_number_max
    df['amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2'] \
        = amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2
    df['days_last_due_prev_app_first5_max_prev_app_first5'] \
        = days_last_due_prev_app_first5_max_prev_app_first5
    df['hour_appr_process_start'] = hour_appr_process_start
    df['sellerplace_area_prev_app_first5_max_prev_app_first5'] \
        = sellerplace_area_prev_app_first5_max_prev_app_first5
    df['days_last_due_prev_app_last5_mean_prev_app_last5'] \
        = days_last_due_prev_app_last5_mean_prev_app_last5
    df['days_instalment_sum'] = days_instalment_sum
    df['amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5'] \
        = amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5
    df['payment_delay_mean'] = payment_delay_mean
    df['payment_difference_mean'] = payment_difference_mean
    df['cnt_payment_prev_app_last5_sum_prev_app_last5'] = cnt_payment_prev_app_last5_sum_prev_app_last5
    df['day_payment_difference_sum'] = day_payment_difference_sum
    df['amt_instalment_max'] = amt_instalment_max
    df['num_instalment_number_mean'] = num_instalment_number_mean
    df['days_last_due_1st_version_prev_app_last5_sum_prev_app_last5'] \
        = days_last_due_1st_version_prev_app_last5_sum_prev_app_last5
        
    df['name_family_status'] = name_family_status
   
    df_preprocessed = preprocess.transform(df)
    
    column_names = ['numeric__ext_source_cols_mean',
       'numeric__amt_credit_amt_annuity_ratio',
       'numeric__days_birth',
       'numeric__ext_source_3', 
       'numeric__amt_annuity',
       'numeric__days_credit_enddate_first',
       'numeric__ext_source_2',
       'numeric__days_employed', 
       'numeric__ext_source_1',
       'numeric__days_id_publish', 
       'numeric__amt_credit_sum_sum',
       'numeric__days_registration', 
       'numeric__amt_credit_sum_debt_sum',
       'numeric__amt_credit', 
       'numeric__days_enddate_fact_max',
       'numeric__days_credit_update_min',
       'numeric__cnt_instalment_future_pos_first2_sum_pos_first2',
       'numeric__amt_credit_sum_max',
       'numeric__region_population_relative', 
       'numeric__own_car_age',
       'numeric__days_instalment_max', 
       'numeric__amt_income_total',
       'categorical__organization_type',
       'categorical__name_education_type', 
       'categorical__code_gender',
       'numeric__days_last_phone_change',
       'numeric__day_payment_difference_min',
       'numeric__amt_credit_max_overdue_max',
       'numeric__amt_instalment_first',
       'numeric__day_payment_difference_pay_first5_sum_pay_first5',
       'numeric__day_payment_difference_first',
       'numeric__amt_annuity_prev_app_first2_mean_prev_app_first2',
       'numeric__amt_instalment_sum', 
       'numeric__year_balance_first_first',
       'numeric__days_last_due_prev_app_last5_max_prev_app_last5',
       'numeric__days_first_due_prev_app_first5_sum_prev_app_first5',
       'numeric__day_payment_difference_pay_first2_sum_pay_first2',
       'numeric__day_payment_difference_pay_last5_sum_pay_last5',
       'numeric__apartments_avg',
       'numeric__amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5',
       'numeric__days_instalment_min',
       'numeric__day_payment_difference_max',
       'numeric__days_last_due_1st_version_prev_app_first5_sum_prev_app_first5',
       'numeric__amt_instalment_pay_last5_max_pay_last5',
       'numeric__days_instalment_mean',
       'numeric__amt_credit_sum_limit_max',
       'numeric__num_instalment_version_sum', 
       'numeric__basementarea_avg',
       'numeric__days_first_due_prev_app_first5_max_prev_app_first5',
       'numeric__cnt_instalment_pos_first2_sum_pos_first2',
       'numeric__status_mean_mean',
       'numeric__years_beginexpluatation_avg',
       'numeric__day_payment_difference_mean',
       'numeric__amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5',
       'numeric__num_instalment_number_max',
       'numeric__amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2',
       'numeric__days_last_due_prev_app_first5_max_prev_app_first5',
       'numeric__hour_appr_process_start',
       'numeric__sellerplace_area_prev_app_first5_max_prev_app_first5',
       'numeric__days_last_due_prev_app_last5_mean_prev_app_last5',
       'numeric__days_instalment_sum',
       'numeric__amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5',
       'numeric__payment_delay_mean', 
       'numeric__payment_difference_mean',
       'numeric__cnt_payment_prev_app_last5_sum_prev_app_last5',
       'numeric__day_payment_difference_sum',
       'numeric__amt_instalment_max',
       'numeric__num_instalment_number_mean',
       'numeric__days_last_due_1st_version_prev_app_last5_sum_prev_app_last5',
       'categorical__name_family_status']
       
    
    data = df_preprocessed.loc[:, column_names]
    
    return data

st.title('Loan default prediction')
st.header('Applicant data:')

# Categorical 
code_gender = st.selectbox('gender', ['M', 'F', 'XNA'])
name_education_type = st.selectbox('name_education_type', 
                                   ['Secondary / secondary special', 
                                    'Higher education', 'Incomplete higher', 
                                    'Lower secondary', 'Academic degree'])

organization_type = st.selectbox('organization_type', 
                                 ['Self-employed', 'XNA', 'Trade: type 3',
                                 'Business Entity Type 3', 
                                'Construction', 'Other', 'Kindergarten', 
                                'Trade: type 7', 'Security Ministries', 
                                'Industry: type 11', 'Police', 'School',
                                'Business Entity Type 2', 'Medicine', 
                                'Business Entity Type 1', 'Agriculture',
                                'Legal Services', 'Security', 'Government',
                                'Transport: type 2', 'Military', 
                                'Industry: type 3', 'Industry: type 2', 
                                'Industry: type 9', 'Restaurant', 'Telecom',
                                'Housing', 'Emergency', 'Trade: type 2', 'Bank',
                                'Industry: type 7', 'Mobile', 'University', 
                                'Insurance', 'Transport: type 4', 'Postal', 
                                'Hotel', 'Transport: type 3',
                                'Services', 'Industry: type 1', 
                                'Industry: type 4', 'Industry: type 13', 
                                'Cleaning', 'Electricity', 'Realtor',
                                'Trade: type 6', 'Transport: type 1', 
                                'Advertising', 'Industry: type 5', 
                                'Industry: type 6', 'Industry: type 12',
                                'Trade: type 1', 'Culture', 'Trade: type 5',
                                'Religion', 'Industry: type 10',
                                'Trade: type 4', 'Industry: type 8'])

name_family_status = st.selectbox('name_family_status', ['Married', 'Civil marriage',
                                                       'Single / not married',
                                                       'Widow', 'Separated',
                                                       'Unknown'])

# Numeric                 
ext_source_cols_mean = st.number_input('ext_source_cols_mean', min_value=0.0, 
                                       max_value=1.0, value=0.386565)

amt_credit_amt_annuity_ratio = st.number_input('amt_credit_amt_annuity_ratio', 
                                               min_value=1.0, 
                                               max_value=50.0, value=25.651473)
days_birth = st.number_input('days_birth', min_value=-25200, 
                             max_value=-6000, value=-11442)

ext_source_3 = st.number_input('ext_source_3', min_value=0.0, 
                               max_value=1.0, value=0.244155)

amt_annuity = st.number_input('amt_annuity', min_value=1500.0, 
                               max_value=300000.0, value=21226.5)

days_credit_enddate_first = st.number_input('days_credit_enddate_first', 
                                            min_value=-17000.0, 
                                            max_value=32000.0, 
                                            value=389.0)

ext_source_2 = st.number_input('ext_source_2', min_value=0.0, 
                               max_value=1.0, value=0.328976)

days_employed = st.number_input('days_employed', min_value=-18000.0, 
                                max_value=0.0, value=-622.0)


ext_source_1 = st.number_input('ext_source_1', min_value=0.0, 
                               max_value=1.0, value=0.3044)

days_id_publish = st.number_input('days_id_publish', min_value=-7000, 
                                  max_value=0, value=-3957)

amt_credit_sum_sum = st.number_input('amt_credit_sum_sum', 
                                     min_value=0.0, 
                                  max_value=481622517.46, value=857416.005)

days_registration = st.number_input('days_registration', 
                                    min_value=-25672.0, 
                                    max_value=0.0,
                                    value=-5752.0)

amt_credit_sum_debt_sum = st.number_input('amt_credit_sum_debt_sum', 
                                            min_value=-3112461.135, 
                                           max_value=334498331.20500004, 
                                           value=227029.68)

amt_credit = st.number_input('amt_credit', min_value=450000.0,
                             max_value=4050000.0, value=544491.0)

days_enddate_fact_max = st.number_input('days_enddate_fact_max', 
                                    min_value=-3000.0, 
                                    max_value=0.0,
                                    value=-150.0) 

days_credit_update_min = st.number_input('days_credit_update_min', 
                                    min_value=-3000.0, 
                                    max_value=0.0,
                                    value=-1152.0)

cnt_instalment_future_pos_first2_sum_pos_first2 \
    = st.number_input('cnt_instalment_future_pos_first2_sum_pos_first2', 
                      min_value=0.0, 
                      max_value=140.0, value=4.0)

amt_credit_sum_max = st.number_input('amt_credit_sum_max', min_value=0.0,
                             max_value=180100000.0, value=324000.0)

region_population_relative = st.number_input('region_population_relative', 
                                             min_value=0.0,
                                             max_value=1.0, 
                                             value=0.025164)

own_car_age = st.number_input('own_car_age', min_value=0.0,
                             max_value=100.0, value=8.0)

days_instalment_max = st.number_input('days_instalment_max', 
                                    min_value=-3922.0, 
                                    max_value=0.0,
                                    value=-148.0) 

amt_income_total = st.number_input('amt_income_total', 
                                    min_value=0.0, 
                                    max_value=120000000.0,
                                    value=270000.0)


days_last_phone_change = st.number_input('days_last_phone_change', 
                                    min_value=-5000.0, 
                                    max_value=0.0,
                                    value=-372.0) 

day_payment_difference_min = st.number_input('day_payment_difference_min',
                                             min_value=-3000.0, max_value=160.0,
                                             value=9.0)

amt_credit_max_overdue_max = st.number_input('amt_credit_max_overdue_max', 
                                             min_value=0.0,
                                             max_value=135987185.0, 
                                             value=0.0)

amt_instalment_first = st.number_input('amt_instalment_first', 
                                        min_value=0.0,
                                        max_value=3971487.845, 
                                        value=15015.555)

day_payment_difference_pay_first5_sum_pay_first5 \
     = st.number_input('day_payment_difference_pay_first5_sum_pay_first5', 
                       min_value=0.0, 
                       max_value=5000.0, 
                       value=169.0) 

day_payment_difference_first = st.number_input('day_payment_difference_first', 
                                    min_value=-3000.0, 
                                    max_value=500.0,
                                    value=9.0) 


amt_annuity_prev_app_first2_mean_prev_app_first2 \
    = st.number_input('amt_annuity_prev_app_first2_mean_prev_app_first2', 
                        min_value=0.0, 
                        max_value=3888.27,
                        value=3888.27)
    
amt_instalment_sum = st.number_input('amt_instalment_sum', min_value=0.0, 
                                  max_value=17941758.355, value=38345.175)

year_balance_first_first = st.number_input('year_balance_first_first', 
                                    min_value=0.0, 
                                    max_value=10.0,
                                    value=0.416667)

days_first_due_prev_app_first5_sum_prev_app_first5 \
    = st.number_input('days_first_due_prev_app_first5_sum_prev_app_first5', 
                        min_value=-12873.0, 
                        max_value=0.0,
                        value=-328.0) 

days_last_due_prev_app_last5_max_prev_app_last5 \
    = st.number_input('days_last_due_prev_app_last5_max_prev_app_last5', 
                        min_value=-3000.0, 
                        max_value=0.0,
                        value=-148.0)  

day_payment_difference_pay_first2_sum_pay_first2 \
    = st.number_input('day_payment_difference_pay_first2_sum_pay_first2', 
                        min_value=-5000.0, 
                        max_value=1000.00,
                        value=49.0) 

day_payment_difference_pay_last5_sum_pay_last5 \
     = st.number_input('day_payment_difference_pay_last5_sum_pay_last5', 
                         min_value=-3000.0, 
                         max_value=2000.00,
                         value=187.0) 

apartments_avg = st.number_input('apartments_avg', 
                        min_value=0.0, 
                        max_value=1.0,
                        value=0.0876) 

amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5 \
    = st.number_input('amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5', 
                        min_value=0.0,
                        max_value=10.0, 
                        value=1.0)

days_instalment_min = st.number_input('days_instalment_min', 
                                        min_value=-3000.0, 
                                        max_value=0.0,
                                        value=-328.0)  

day_payment_difference_max = st.number_input('day_payment_difference_max', 
                                    min_value=-50.0, 
                                    max_value=2000.0,
                                    value=42.0)

days_last_due_1st_version_prev_app_first5_sum_prev_app_first5 \
    = st.number_input('days_last_due_1st_version_prev_app_first5_sum_prev_app_first5', 
                            min_value=-20000.0, 
                            max_value=5000.0,
                            value=-58.0) 

amt_instalment_pay_last5_max_pay_last5 \
    = st.number_input('amt_instalment_pay_last5_max_pay_last5', 
                            min_value=0.0, 
                            max_value=3473582.895,
                            value=3888.27)

days_instalment_mean = st.number_input('days_instalment_mean', 
                                    min_value=-3000.0, 
                                    max_value=0.0,
                                    value=-238.0) 

amt_credit_sum_limit_max = st.number_input('amt_credit_sum_limit_max', 
                                            min_value=0.0, 
                                            max_value=4500000.0,
                                            value=37981.305)

num_instalment_version_sum = st.number_input('num_instalment_version_sum', 
                                                min_value=0.0, 
                                                max_value=2000.0,
                                                value=9.0)    

basementarea_avg = st.number_input('basementarea_avg', 
                                    min_value=0.0, 
                                    max_value=1.0,
                                    value=0.0762)

days_first_due_prev_app_first5_max_prev_app_first5 \
    = st.number_input('days_first_due_prev_app_first5_max_prev_app_first5', 
                                        min_value=-3000.0, 
                                        max_value=-10.0,
                                        value=-328.0)

cnt_instalment_pos_first2_sum_pos_first2 \
    = st.number_input('cnt_instalment_pos_first2_sum_pos_first2', 
                                        min_value=0.0, 
                                        max_value=150.0,
                                        value=17.0)

status_mean_mean = st.number_input('status_mean_mean', 
                                    min_value=0.0, 
                                    max_value=150.0,
                                    value=1.39) 

years_beginexpluatation_avg = st.number_input('years_beginexpluatation_avg', 
                                                min_value=0.0, 
                                                max_value=1.0,
                                                value=0.9816) 

day_payment_difference_mean \
    = st.number_input('day_payment_difference_mean', 
                        min_value=-800.0, 
                        max_value=300.0,
                        value=33.714286)

amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5 \
    = st.number_input('amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5', 
                        min_value=0.0, 
                        max_value=10.0,
                        value=1.0) 

num_instalment_number_max = st.number_input('num_instalment_number_max', 
                                            min_value=0.0, 
                                            max_value=500.0,
                                            value=7.0)

amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2 \
    = st.number_input('amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2', 
                        min_value=0.0, 
                        max_value=14.0,
                        value=1.0)

days_last_due_prev_app_first5_max_prev_app_first5 \
    = st.number_input('days_last_due_prev_app_first5_max_prev_app_first5', 
                                    min_value=-3000.0, 
                                    max_value=0.0,
                                    value=-148.0)

hour_appr_process_start \
    = st.number_input('hour_appr_process_start', 
                        min_value=0, 
                        max_value=23,
                        value=14) 

sellerplace_area_prev_app_first5_max_prev_app_first5 \
    = st.number_input('sellerplace_area_prev_app_first5_max_prev_app_first5', 
                        min_value=-1.0, 
                        max_value=250000.0,
                        value=15.0)

days_last_due_prev_app_last5_mean_prev_app_last5 \
    = st.number_input('days_last_due_prev_app_last5_mean_prev_app_last5', 
                                            min_value=-3000.0,
                                            max_value=-21.0, 
                                            value=-148.0)

days_instalment_sum = st.number_input('days_instalment_sum', 
                                    min_value=-3000.0, 
                                    max_value=0.0,
                                    value=-1666.0)

amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5 \
    = st.number_input('amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5', 
                                            min_value=0.0,
                                            max_value=30.0, 
                                            value=1.0)

payment_delay_mean = st.number_input('payment_delay_mean', 
                                    min_value=0.0, 
                                    max_value=1.0,
                                    value=0.0)

payment_difference_mean \
    = st.number_input('payment_difference_mean', 
                        min_value=-357496.805, 
                        max_value=156145.9,
                        value=0.0)

cnt_payment_prev_app_last5_sum_prev_app_last5 \
    = st.number_input('cnt_payment_prev_app_last5_sum_prev_app_last5', 
                        min_value=0.0, 
                        max_value=500.0,
                        value=10.0)

day_payment_difference_sum = st.number_input('day_payment_difference_sum', 
                                    min_value=-54706.0, 
                                    max_value=10529.0,
                                    value=236.0)


amt_instalment_max = st.number_input('amt_instalment_max', 
                                    min_value=0.0,
                                    max_value=3971487.845, 
                                    value=15015.555)

num_instalment_number_mean \
    = st.number_input('num_instalment_number_mean', 
                        min_value=0.0, 
                        max_value=150.0,
                        value=4.0)

days_last_due_1st_version_prev_app_last5_sum_prev_app_last5 \
    = st.number_input('days_last_due_1st_version_prev_app_last5_sum_prev_app_last5', 
                        min_value=-15000.0, 
                        max_value=350.0,
                        value=-58.0)

if st.button('Predict loan default'):
    data = prepare_data(ext_source_cols_mean,
       amt_credit_amt_annuity_ratio, days_birth,
       ext_source_3, amt_annuity,
       days_credit_enddate_first, 
       ext_source_2, days_employed, ext_source_1,
       days_id_publish, amt_credit_sum_sum,
       days_registration, amt_credit_sum_debt_sum,
       amt_credit, days_enddate_fact_max,
       days_credit_update_min,
       cnt_instalment_future_pos_first2_sum_pos_first2,
       amt_credit_sum_max,
       region_population_relative,
       own_car_age,
       days_instalment_max, amt_income_total,
       organization_type,
       name_education_type, code_gender,
       days_last_phone_change,
       day_payment_difference_min,
       amt_credit_max_overdue_max,
       amt_instalment_first, 
       day_payment_difference_pay_first5_sum_pay_first5,
       day_payment_difference_first,
       amt_annuity_prev_app_first2_mean_prev_app_first2,
       amt_instalment_sum, year_balance_first_first,
       days_last_due_prev_app_last5_max_prev_app_last5,
       days_first_due_prev_app_first5_sum_prev_app_first5,
       day_payment_difference_pay_first2_sum_pay_first2,
       day_payment_difference_pay_last5_sum_pay_last5,
       apartments_avg,
       amt_goods_price_amt_credit_ratio_prev_app_last5_mean_prev_app_last5,
       days_instalment_min,
       day_payment_difference_max,
       days_last_due_1st_version_prev_app_first5_sum_prev_app_first5,
       amt_instalment_pay_last5_max_pay_last5,
       days_instalment_mean,
       amt_credit_sum_limit_max,
       num_instalment_version_sum, 
       basementarea_avg,
       days_first_due_prev_app_first5_max_prev_app_first5,
       cnt_instalment_pos_first2_sum_pos_first2,
       status_mean_mean,
       years_beginexpluatation_avg,
       day_payment_difference_mean,
       amt_application_amt_credit_diff_prev_app_last5_mean_prev_app_last5,
       num_instalment_number_max,
       amt_application_amt_credit_diff_prev_app_last2_sum_prev_app_last2,
       days_last_due_prev_app_first5_max_prev_app_first5,
       hour_appr_process_start,
       sellerplace_area_prev_app_first5_max_prev_app_first5,
       days_last_due_prev_app_last5_mean_prev_app_last5,
       days_instalment_sum,
       amt_application_amt_credit_diff_prev_app_last5_sum_prev_app_last5,
       payment_delay_mean, 
       payment_difference_mean,
       cnt_payment_prev_app_last5_sum_prev_app_last5,
       day_payment_difference_sum,
       amt_instalment_max,
       num_instalment_number_mean,
       days_last_due_1st_version_prev_app_last5_sum_prev_app_last5,
       name_family_status)
    
    default_prediction = make_prediction(default_model, data, threshold)
    st.success(default_prediction)

