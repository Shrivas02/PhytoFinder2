import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Load your trained model
model = tf.keras.models.load_model('phytofinder.keras')

# Class labels
class_names = ['neem', 'tulsi']  # Update this as needed

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://fir-b0e7f-default-rtdb.firebaseio.com/'


# Firebase DB reference
        # Load Firebase credentials from secrets (for Streamlit Cloud)
firebase_creds = st.secrets["firebase"]
cred = credentials.Certificate(firebase_creds)

# Initialize Firebase Admin SDK
firebase_admin.initialize_app(cred)
ref = db.reference('/plant_medicinal_data')
data = ref.get()


# Streamlit UI
st.title("🌿 PhytoFinder")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🌱 Identified as: **{predicted_class}**")

    # Firebase lookup
    plant_name = predicted_class.lower()
    st.write("🧪 Predicted class:", predicted_class)
    st.write("🔍 Plant name used for Firebase:", plant_name)
    st.write("📋 Firebase keys found:", list(data.keys()))

    plant_data = data.get(plant_name)

    if plant_data:
        st.subheader("Medicinal Information")
        for key, value in plant_data.items():
            st.markdown(f"**{key}**")
            if isinstance(value, list):
                for item in value:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"{value}")
    else:
        st.warning(f"No medicinal info found for: `{plant_name}`")
        st.write("Available keys in Firebase:", list(data.keys()))
