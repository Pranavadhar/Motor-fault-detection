import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def fault_predictor(mean, variance, kurtosis):
    df = pd.read_csv('final_feat_xtract.csv')
    X = df[['Mean', 'Variance', 'Kurtosis']]
    y = df['Condition']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    user_input_df = pd.DataFrame(
        {'Mean': [mean], 'Variance': [variance], 'Kurtosis': [kurtosis]})
    prediction = model.predict(user_input_df)
    return prediction[0]

st.title("MACHINE CONDITION DETECTION - AN EDSP END SEM PROJECT")

mean_input = st.number_input("Mean")
variance_input = st.number_input("Variance")
kurtosis_input = st.number_input("Kurtosis")

if st.button("Predict"):
    prediction = fault_predictor(mean_input, variance_input, kurtosis_input)
    st.write(f"Condition of the Machine: {prediction}")
