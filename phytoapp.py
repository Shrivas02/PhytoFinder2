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

    # Fetch medicinal info from Firebase

# if not firebase_admin._apps:
#     cred = credentials.Certificate(dict(st.secrets["firebase"]))
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://your-project-id.firebaseio.com/'
#     })
# ref = db.reference('/plant_medicinal_data')

# # Reading data
# data = ref.get()

# # Writing data
# ref.push({
#     'plant': 'tulsi',
#     'uses': ['cold', 'fever']
# })                  
