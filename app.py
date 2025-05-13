import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import joblib

# dataset = https://frenzy86.s3.eu-west-2.amazonaws.com/fav/iris.data

def main():
    st.title('prima app')
    url = st.text_input('Incolla dataset url:',"https://frenzy86.s3.eu-west-2.amazonaws.com/fav/iris.data")
    df = pd.read_csv(url,header=None)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    df1 = df.copy()
    st.dataframe(df1)

    st.header(":abacus: Statistica")
    st.dataframe(df1.describe())

    st.header(":chart_with_upwards_trend: Pairplot")
        
    fig = plt.figure(figsize=(12,12))
    fig = sns.pairplot(df1, hue='class',  height=1.5, palette="viridis")
    st.pyplot(fig)

    st.header(":star: Correlazione")
        
    df_numeric = df1.select_dtypes(include = ['float', 'int'])
    corr = df_numeric.corr()

    fig2,ax = plt.subplots(figsize = (8,6))
    sns.heatmap(corr, annot=True, cmap = 'Blues', ax=ax)
    st.pyplot(fig2)

    st.header(":magic_wand: :rainbow[Inferenza]")

    sepal_length = st.number_input('sepal length', min_value = 2.0, max_value = 9.0)
    sepal_width = st.number_input('sepal width', min_value = 0.1, max_value = 5.0)
    petal_length = st.number_input('petal length', min_value = 0.1, max_value = 8.0)
    petal_width = st.number_input('petal width', min_value = 0.1, max_value = 4.0)

    data = {
        "sepal length": [sepal_length],
        "sepal width": [sepal_width],
        "petal length": [petal_length],
        "petal width": [petal_width]
            }
        
    pred_data = pd.DataFrame(data)

    iris_pipe = joblib.load('iris_pipe.pkl')
        
    if st.button(':crystal_ball: Fare predizione'):
        pred = iris_pipe.predict(pred_data)[0]          # con [0] me da solo el elemento, la stringa. Si no, me daria el array!
        st.success(f":cherry_blossom: La predizione corretta è: {pred} :cherry_blossom:")


if __name__ == "__main__":
    main()