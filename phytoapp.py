# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://your-project-id.firebaseio.com/'
    })

ref = db.reference('/plant_medicinal_data')
data = ref.get()  # Read once here

# Streamlit UI
st.title("🌿 PhytoFinder")
uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    ...
    # all your image code here...

    plant_data = data.get(predicted_class.lower())

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
        st.warning("No medicinal data found.")
