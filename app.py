import streamlit as st
import numpy as np
from page1 import show_identifier_page
from examples import example1, example2
from page2 import camera_input


options = st.sidebar.radio("Test it out!", ('Example 1', 'Example 2', 'Upload Image', 'Take a photo'))

if options == 'Upload Image':
	show_identifier_page()
elif options == 'Example 1':
	example1()
elif options == 'Example 2':
	example2()
elif options == 'Take a photo':
	camera_input()


