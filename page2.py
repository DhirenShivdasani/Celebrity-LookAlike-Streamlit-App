import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import pandas as pd
from CelebLookAlike import *
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE



def camera_input():
	st.title('Find your Celebrity Lookalike!')

	name = st.text_input('Enter a name for this picture')
	file_image = st.camera_input(label = "Take a picture!")

	if file_image:
		input_image = Image.open(file_image)

		predict = st.button('Find celebrity lookalike')
		all_embeds = pd.read_csv('img_embeddings134.csv', index_col = [0])

		filename = f'{name}.png'
		img = input_image.save(filename)

		if predict:
			df = make_distances_df(filename, all_embeds)
			with st.spinner('Identifying celebrity lookalike'):
				time.sleep(1)
			st.success('Done!')

			st.subheader('Your celebrity lookalike is ' + df.iloc[0]['celeb'])
			

			uploaded_img = load_img(f'{name}.png')
			mtcnn_img = img_to_mtcnn(uploaded_img)
			celeb_look_alike_img = get_celeb_look_alike_img(df, all_embeds)
			embed = mtcnn_to_embedding(mtcnn_img)
			embed_img = show_embed(embed)

			st.image([cv2.resize(uploaded_img, (150,200)), cv2.resize(load_mtcnn(mtcnn_img), (150,200)), cv2.resize(embed_img, (150,200), interpolation = cv2.INTER_AREA), cv2.resize(celeb_look_alike_img, (150,200))], clamp = True)

			st.title('Process')
			st.caption("Using MTCNN, a pre-trained facial recognition model, I was able to extract just the cropped face of the uploaded image through identification of the facial features (i.e. two eyes, nose, and endpoints of the mouth). Now that I have a cropped face of the uploaded image, I used the InceptionResnetV1(pretrained='vggface2') that calculates the face embedding that represents the features extracted from the cropped face.\n\n To classify this image I needed to find a way to compare the face embeddings of the uploaded image to the face embeddings of the celebrities (the face embedding for the uploaded image is shown above). I accomplished this by calculating Euclidean Distance between the uploaded face embedding and each celebrity's face embedding. What this tells us is that the smaller the distance, the more similar those images are.\n\n So, I made a data frame with all the Euclidean Distance values and included the names of the celebrities assigned to these values to organize all the data. I sorted this data frame in ascending order so the smallest distance would appear first in the data frame and then I made a barplot to display these values (the barplot and data frame is shown below). Based off these values, the celebrity that looks most like the uploaded image is the one that appears first in the data frame and is the first bar in the barplot.")
			st.title('Visuals')	



			st.write(df)

			barplot = make_barplot(df)

			st.pyplot(barplot)
			

			tsne = make_plot(filename, all_embeds)
			st.pyplot(tsne)

			os.remove(filename)









