import pandas as pd
import gradio as gr
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


iface = gr.Interface(fn=fault_predictor,
                     inputs=["number", "number", "number"],
                     outputs=gr.Textbox(label="Condition of the Machine"),
                     live=True,
                     title="MACHINE CONDITION DETECTION - AN EDSP END SEM PROJECT",
                     description="This is an END to END EMBEDDED DIGITAL SIGNAL PROCESSING(EDSP) project done to predict the condition of the motor by giving the inputs in the prompt. \n\n"
                     "This fault detection project has been deployed to showcase the main objective of the condition of the machine whether it is in a healthy or in an unhealthy condition. \n\n"
                     "DEPLOYMENT TOOL: GRADIO \n\n"
                     "HOST: HUGGING FACE \n\n")
iface.launch(share=True)
