import tkinter as tk
from tkinter import ttk, BOTH
from tkinter.ttk import Entry

import pandas as pd
from joblib import load
import numpy as np

# Load saved machine learning models
models = ['RandomForest', 'SVM', 'KNN', 'LightGBM', 'Ensemble']
model_files = ['rfc_model.pkl', 'svc_model.pkl', 'knn_model.pkl', 'lgbm_model.pkl', 'ensemble_model.pkl']
loaded_models = [load(model) for model in model_files]


# Define GUI
class MLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Machine Learning App')

        self.frame_1 = ttk.Frame(self)
        self.frame_1.pack()

        # Model selection dropdown
        self.model_var = tk.StringVar()
        self.model_var.set(models[0])
        self.model_dropdown = ttk.OptionMenu(self.frame_1, self.model_var, models[0], *models)
        self.model_dropdown.pack()

        # # Input fields
        self.feature_names = ['CreditScore',
                              'Gender',
                              'Age',
                              'Tenure',
                              'Balance',
                              'NumOfProducts',
                              'HasCrCard',
                              'IsActiveMember',
                              'EstimatedSalary', ]
        self.input_var = tk.StringVar()
        label = tk.Label(self.frame_1, text=' '.join(str(e) for e in self.feature_names))
        entry = tk.Entry(self.frame_1, textvariable=self.input_var)
        label.pack()
        entry.pack(fill=BOTH, expand=1)

        # Prediction button
        self.predict_button = tk.Button(self.frame_1, text='Predict', command=self.predict)
        self.predict_button.pack()

        # Prediction output
        self.output_label = tk.Label(self.frame_1, text='')
        self.output_label.pack()

    # Predict function for button command
    def predict(self):
        # Get selected model index and make prediction
        model_index = models.index(self.model_var.get())
        model = loaded_models[model_index]

        # Get input values as numpy array
        # input_values = np.array([float(var.get()) for var in self.input_vars])
        input_values = [float(st) for st in self.input_var.get().split('	')]

        dic = {self.feature_names[i]: input_values[i] for i in range(len(self.feature_names))}
        df = pd.DataFrame(dic, index=[0])

        prediction = model.predict(df)[0]

        # Update output label with prediction
        self.output_label.config(text=f'Prediction: {prediction}')
        self.output_label.config(text=f'Prediction: Exited - {False if prediction == 0 else True}')


# Create and run app
app = MLApp()
app.mainloop()
