import streamlit as st
import cv2
import numpy as np
import joblib

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'SVM': 'svm_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    # 'Linear Regression': 'linear_regression_model.pkl'
}

CLASS_LABELS = ['Covid', 'Normal', 'Viral Pneumonia']

st.set_page_config(page_title='COVID-19 Classifier', layout='centered')
st.title('COVID-19 Chest Classification')
st.write('Upload a chest image and select a model to classify it as Covid, Normal, or Viral Pneumonia.')

st.sidebar.header('Model Selection')
selected_model = st.sidebar.selectbox('Choose a model', list(MODEL_FILES.keys()))

@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_FILES[model_name])

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (64, 64))
    flattened = image_resized.flatten() / 255.0
    return image, flattened

model = load_model(selected_model)

uploaded_file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image, features = preprocess_image(uploaded_file)

    prediction = model.predict([features])[0]
    raw_prediction = prediction

    predicted_label = CLASS_LABELS[prediction]

    st.image(image, caption='Uploaded Image', width=400)
    st.markdown(f'Prediction: **{predicted_label}**')

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([features])[0]
        st.markdown('Predicted Probabilities')
    else:
        probabilities = None

    if probabilities is not None:
        for label, prob in zip(CLASS_LABELS, probabilities):
            st.write(f'- **{label}:** {prob:.4f}')
