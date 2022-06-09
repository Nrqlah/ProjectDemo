import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

col1, col2, col3 = st.columns(3)
with col1:
    image = Image.open('Iris_virginica_2.jpg')
    st.image(image,'Iris Virginica Species')
with col2:
    image = Image.open('IRIS_VERSICOLOR.jpg')
    st.image(image,'Iris Versicolor Species')
with col3:
    image = Image.open('iris_setosa.jpg')
    st.image(image,'Iris Setosa Species')

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = pd.read_csv('https://raw.githubusercontent.com/Nrqlah/ProjectDemo/main/IRIS.csv')
X = iris.drop(['species'],axis=1)
Y = iris['species']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
target = ['setosa','versicolor','virginica']
st.dataframe({'Species':target})

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
