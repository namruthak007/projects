import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model('mymodel.h5')
columns = ['laufkont', 'laufzeit', 'moral', 'verw', 'hoehe', 'sparkont', 'beszeit', 'rate',
           'famges', 'buerge', 'wohnzeit', 'verm', 'alter', 'weitkred', 'Wohn',
           'bishkred', 'beruf', 'pers', 'telef', 'gastarb']

numerical_cols = ['laufzeit', 'hoehe', 'beszeit', 'rate', 'alter', 'weitkred', 'bishkred', 'pers']

scaler = StandardScaler()
scaler.fit(np.random.randn(100, len(numerical_cols)))

st.title("Credit Risk Prediction")

input_data = []
for col in columns:
    value = st.number_input(f"Enter value for {col}:", min_value=0)
    input_data.append(value)

custom_input_df = pd.DataFrame([input_data], columns=columns)

custom_input_df[numerical_cols] = scaler.transform(custom_input_df[numerical_cols])

if st.button('Predict'):
    custom_prediction = model.predict(custom_input_df)
    custom_prediction = (custom_prediction > 0.5).astype(int)

    if custom_prediction[0][0] == 1:
        st.success('Prediction: Good Credit')
    else:
        st.error('Prediction: Bad Credit')
