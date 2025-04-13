import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import json

# Load model
model = tf.keras.models.load_model('phytofinder.keras')

# Class names
class_names = ['neem', 'tulsi']  # Update as needed

# Load plant data from JSON
with open('Plants_data.json', 'r') as f:
    plant_info = json.load(f)

# Streamlit UI
st.title("🌿 PhytoFinder")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    plant_name = predicted_class.lower()

    st.markdown(f"### 🌱 Identified as: **{plant_name}**")

    # Show info from JSON
    if plant_name in plant_info:
        st.subheader("Medicinal Information")
        for key, value in plant_info[plant_name].items():
            st.markdown(f"**{key}**")
            if isinstance(value, list):
                for item in value:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"{value}")
    else:
        st.warning(f"No medicinal info found for: `{plant_name}`")
        st.write("Available keys:", list(plant_info.keys()))
