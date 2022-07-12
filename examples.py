import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import pandas as pd
from CelebLookAlike import *
import seaborn as sns
from sklearn.manifold import TSNE



def example1():
	example_img = load_img('Test/genericface2.png')
	st.image(example_img)

	test = st.button('Find celebrity lookalike')
	all_embeds = pd.read_csv('img_embeddings134.csv', index_col = [0])
	df = make_distances_df('Test/genericface2.png', all_embeds)

	if test:
		mtcnn_img = img_to_mtcnn(example_img)
		# st.image(load_mtcnn(mtcnn_img), clamp = True)
		celeb_look_alike_img = get_celeb_look_alike_img(df, all_embeds)
		# st.image([cv2.resize(celeb_look_alike_img, (200,200)), cv2.resize(load_mtcnn(mtcnn_img), (200,200))], clamp = True)
		with st.spinner('Identifying celebrity lookalike'):
		    time.sleep(2)
		st.success('Done!')
		st.subheader(df.iloc[0]['celeb'])
		st.image([cv2.resize(celeb_look_alike_img, (200,200)), cv2.resize(load_mtcnn(mtcnn_img), (200,200))], clamp = True)
	

def example2():
	example_img = load_img('Test/woman2.jpeg')
	st.image(example_img)

	test = st.button('Find celebrity lookalike')
	all_embeds = pd.read_csv('img_embeddings134.csv', index_col = [0])
	df = make_distances_df('Test/woman2.jpeg', all_embeds)

	if test:
		mtcnn_img = img_to_mtcnn(example_img)
		# st.image(load_mtcnn(mtcnn_img), clamp = True)
		celeb_look_alike_img = get_celeb_look_alike_img(df, all_embeds)
		# st.image([cv2.resize(celeb_look_alike_img, (200,200)), cv2.resize(load_mtcnn(mtcnn_img), (200,200))], clamp = True)
		with st.spinner('Identifying celebrity lookalike'):
		    time.sleep(2)
		st.success('Done!')
		st.subheader(df.iloc[0]['celeb'])
		st.image([cv2.resize(celeb_look_alike_img, (200,200)), cv2.resize(load_mtcnn(mtcnn_img), (200,200))], clamp = True)
	
