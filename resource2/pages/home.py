'''
Page to be displayed when entering app
'''

# Streamlit dependencies
import streamlit as st
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        st.write(
            "Welcome page with some details on what this app is about")