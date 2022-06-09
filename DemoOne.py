import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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

from PIL import Image
image = 'https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.fs.fed.us%2Fwildflowers%2Fbeauty%2Firis%2FBlue_Flag%2Fimages%2Firis_virginica%2Firis_virginica_virginica_lg.jpg&imgrefurl=https%3A%2F%2Fwww.fs.fed.us%2Fwildflowers%2Fbeauty%2Firis%2FBlue_Flag%2Firis_virginica.shtml&tbnid=yKIaKHMxIOyAVM&vet=12ahUKEwiGpdu3qJ_4AhWz_DgGHVN-Ap4QMygBegUIARDJAQ..i&docid=lWcM6aTHXEpgAM&w=800&h=596&q=Iris%20virginica&ved=2ahUKEwiGpdu3qJ_4AhWz_DgGHVN-Ap4QMygBegUIARDJAQ'
st.image(image, caption='Iris Virginica Species')

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
