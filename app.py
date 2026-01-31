import gradio as gr
import joblib
import numpy as np

model = joblib.load("linear_regression_model.pkl")

def predict(x):
    x = np.array([[x]])
    return model.predict(x)[0]

gr.Interface(
    fn=predict,
    inputs=gr.Number(label="Input value"),
    outputs=gr.Number(label="Prediction"),
    title="Linear Regression Predictor"
).launch()
