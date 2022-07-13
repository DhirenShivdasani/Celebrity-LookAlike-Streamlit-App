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
	example_img = load_img('Test/vinlookalike.jpeg')
	st.image(cv2.resize(example_img, (300,300)))

	test = st.button('Find celebrity lookalike')
	all_embeds = pd.read_csv('img_embeddings134.csv', index_col = [0])
	example_img_embedding = path_to_embedding('Test/vinlookalike.jpeg')
	df = make_distances_df(example_img_embedding, all_embeds)

	if test:
		mtcnn_img = img_to_mtcnn(example_img)
		celeb_look_alike_img = get_celeb_look_alike_img(df, all_embeds)
		with st.spinner('Identifying celebrity lookalike'):
		    time.sleep(2)
		st.success('Done!')
		st.subheader('Celebrity LookAlike: ' + df.iloc[0]['celeb'])
		st.image([cv2.resize(load_mtcnn(mtcnn_img), (200,200)), cv2.resize(show_embed(example_img_embedding), (200,200), interpolation = cv2.INTER_AREA), cv2.resize(celeb_look_alike_img, (200,200))], caption = ['Extracted Face', 'Latent Representation', 'Celebrity LookAlike'], clamp = True)
	

def example2():
	example_img = load_img('Test/jenniferannistonlookalike.png')
	st.image(cv2.resize(example_img, (300,300)))

	test = st.button('Find celebrity lookalike')
	all_embeds = pd.read_csv('img_embeddings134.csv', index_col = [0])
	example_img_embedding = path_to_embedding('Test/jenniferannistonlookalike.png')
	df = make_distances_df(example_img_embedding, all_embeds)
	
	if test:
		mtcnn_img = img_to_mtcnn(example_img)
		celeb_look_alike_img = get_celeb_look_alike_img(df, all_embeds)
		with st.spinner('Identifying celebrity lookalike'):
		    time.sleep(2)
		st.success('Done!')
		st.subheader('Celebrity LookAlike: ' + df.iloc[0]['celeb'])
		st.image([cv2.resize(load_mtcnn(mtcnn_img), (200,200)), cv2.resize(show_embed(example_img_embedding), (200,200), interpolation = cv2.INTER_AREA), cv2.resize(celeb_look_alike_img, (200,200))], caption = ['Extracted Face', 'Latent Representation', 'Celebrity LookAlike'], clamp = True)
	
