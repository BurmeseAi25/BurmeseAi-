import streamlit as st
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("လက်ရေးဂဏန်းခန့်မှန်းမှု")
st.write("သင်၏ လက်ရေးဂဏန်းပုံကို တင်ပါ။")

@st.cache_resource
def load_my_model():
    return load_model("my_model.h5", compile=False)

model = load_my_model()

def predict(img):
    img = img.resize((8, 8))  # dataset size
    img = img.convert("L")
    img = np.array(img)
    img = 16 - img / 16.0  # Normalize
    img = img.reshape(1, 8, 8, 1)
    prediction = model.predict(img)
    return np.argmax(prediction)

uploaded_file = st.file_uploader("ပုံရွေးပါ", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="သင်တင်သောပုံ", use_column_width=True)
    result = predict(img)
    st.success(f"ခန့်မှန်းချက်: {result}")
