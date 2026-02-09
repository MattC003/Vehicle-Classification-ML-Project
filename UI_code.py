import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

@st.cache_resource
def load_prediction_model():
    model_filepath = r"C:\Users\Yarid\Desktop\archive\ResNet50 Files\final_model_ResNet50_improved_run_v3.keras"
    vehicle_model = tf.keras.models.load_model(model_filepath)
    return vehicle_model

vehicle_model = load_prediction_model()

vehicle_classes = ['Van', 'SUV', 'Pickup', 'Convertible', '4dr', '2dr']

def prepare_images(image):
    resized_image = image.resize((224, 224))
    img_array = np.array(resized_image)
    normalized_array = preprocess_input(img_array)
    batched_array = np.expand_dims(normalized_array, axis=0)
    return batched_array

def analyze_predictions(prediction_scores):
    predicted_label = vehicle_classes[np.argmax(prediction_scores)]
    prediction_confidence = np.max(prediction_scores) * 100
    return predicted_label, prediction_confidence

def show_upload_interface():
    st.title("Vehicle Type Detection")
    st.write("Upload an image, and let AI determine the vehicle type.")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)
        st.image(image_to_process, caption="Uploaded Image", use_container_width=True)
        st.write("Image uploaded successfully!")
        st.session_state["uploaded_image"] = image_to_process
        st.session_state["is_image_uploaded"] = True

def show_results_interface():
    st.title("Detection Results")
    image_to_predict = st.session_state["uploaded_image"]
    st.image(image_to_predict, caption="Detected Vehicle", use_container_width=True)
    processed_image = prepare_images(image_to_predict)
    prediction_scores = vehicle_model.predict(processed_image)
    vehicle_type, confidence_score = analyze_predictions(prediction_scores)
    st.markdown(f"<h2 style='text-align: center;'>Confidence: {confidence_score:.2f}%</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>Vehicle Type: {vehicle_type}</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button("Upload Another Image"):
        st.session_state["uploaded_image"] = None
        st.session_state["view_mode"] = "upload"
        st.session_state["is_image_uploaded"] = False

if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "upload"

if "is_image_uploaded" not in st.session_state:
    st.session_state["is_image_uploaded"] = False

if st.session_state["view_mode"] == "upload":
    show_upload_interface()
    if st.session_state["is_image_uploaded"]:
        if st.button("Show Results"):
            st.session_state["view_mode"] = "results"
elif st.session_state["view_mode"] == "results":
    show_results_interface()
