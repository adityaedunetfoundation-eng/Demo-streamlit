import streamlit as st 
import pickle
import numpy as np
model = pickle.load(open("iris_model.pkl","rb"))

st.title("Flower prediction system ")
sepal_length = st.slider("sepal_length",4.0,8.0,5.5)
sepal_width = st.slider("sepal_width",2.0,8.0,3.5)
petal_length = st.slider("petal_length",1.0,7.0,3.5)
petal_width = st.slider("petal_width",0.1,4.5,2.5)

if st.button("submit"):
    features= np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction = model.predict(features)
    output = ["setosa","versicolor","virginica"]
    st.success(f"the predicted result : {output[prediction[0]]}")
