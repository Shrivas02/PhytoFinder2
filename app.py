import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Load your trained model
model = tf.keras.models.load_model('PhytoFinder.keras')

# Class labels (update these)
class_names = ['neem', 'tulsi']


# Load Firebase credentials from secrets
cred = credentials.Certificate(dict(st.secrets["firebase"]))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-project-id.firebaseio.com/'
})




# Streamlit UI
st.title(" Phytofinder ")

uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])
# Fetch data
ref = db.reference('/plant_medicinal_data')
data = ref.get()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    import streamlit as st

st.title("PhytoFinder 🌿")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Your image processing and prediction logic here
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    # Do prediction, Firebase fetch, etc.


    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"###  Identified as: **{predicted_class}**")
    cred = credentials.Certificate("phytofinder_key.json")

