import numpy as np
import streamlit as st

st.set_page_config(page_title="My First Streamlit App")

st.title("My First Streamlit App")
st.write("Hello World!")
st.write("Array:")
st.write(np.arange(0, 2, 0.2))