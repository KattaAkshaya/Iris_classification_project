import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and prepare dataset
@st.cache_data
def load_data():
    df = pd.read_csv("iris_dataset/Iris.csv")  # Adjust path if needed
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y

# Train models once and cache them
@st.cache_resource
def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    return dt, lr

# 🌸 Show flower image based on prediction
def show_flower_image(species):
    image_paths = {
        "Iris-setosa": "images/setosa.jpg.jpeg",
        "Iris-versicolor": "images/versicolor.jpg",
        "Iris-virginica": "images/virginica.jpg"
    }
    img_path = image_paths.get(species)
    if img_path:
        st.image(img_path, caption=f"{species}", width=200)
    else:
        st.warning("No image available for this species.")

# Main app
def main():
    st.title("🌸 Iris Flower Species Prediction")

    dt_classifier, lr_classifier = train_models()

    st.sidebar.header("Input Features")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width],
    })

    st.subheader("Input Parameters")
    st.write(input_data)

    # Predictions
    dt_pred = dt_classifier.predict(input_data)[0]
    lr_pred = lr_classifier.predict(input_data)[0]

    st.subheader("Predictions")
    st.write(f"🌳 Decision Tree Prediction: **{dt_pred}**")
    show_flower_image(dt_pred)

    st.write(f"📈 Logistic Regression Prediction: **{lr_pred}**")
    show_flower_image(lr_pred)

if __name__ == "__main__":
    main()
