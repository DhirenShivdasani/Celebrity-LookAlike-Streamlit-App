# Celebrity-LookAlike-Streamlit-App
Take a picture of yourself or upload in image of yourself to find out who your celebrity lookalike is! (I included only 134 celebrities so far, but I will add more later on)

https://dhirenshivdasani-celebrity-lookalike-streamlit-app-app-mrbmo1.streamlitapp.com/

(Sometimes streamlit crashes so I inlcuded some images from the website below)

# Options
![options](https://user-images.githubusercontent.com/51030977/179425311-18987c4f-c740-4edb-87b5-4a92d421d5ab.png)


# Celebrity LookAlike Identification 

Uploaded image example

![example1 (2)](https://user-images.githubusercontent.com/51030977/179424880-42de3890-658f-4c85-86b1-c6b0e2bb65a9.png)

# Process

Using MTCNN, a pre-trained facial recognition model, I was able to extract just the cropped face of the uploaded images through identification of their facial features (i.e. two eyes, nose, and endpoints of the mouth). Now that I have a cropped face of the uploaded image, I used the InceptionResnetV1(pretrained='vggface2') to calculate the face embedding, which is just a lower-dimensional representation of the image (this is shown in the third image above).

To classify this image I needed to find a way to compare the face embeddings of the uploaded image to the face embeddings of the celebrities. I accomplished this by calculating Euclidean Distance between the uploaded face embedding and each celebrity face embedding. What this tells us is that the smaller the distance, the more similar those images are. So, I made a data frame with all the Euclidean Distance values and included the names of the celebrities assigned to these values to organize all the data. I sorted this data frame in ascending order so the smallest distance would appear first in the data frame and then I made a barplot to display these values (the barplot and data frame are shown below). Based on these values, the celebrity that looks most like the uploaded image is the one that appears first in the data frame and is the first bar in the barplot.

I also included a T-SNE plot, which is a method for visualizing high dimensional data, at the bottom because I wanted to be able to visualize the image embeddings for each celebrity as well as your own.

# Visualizations

![barplot](https://user-images.githubusercontent.com/51030977/179425402-4bff23f9-c6aa-4e75-9556-5b7da189caf4.png)

![tsne](https://user-images.githubusercontent.com/51030977/179425114-47165de2-cf1f-4256-b103-53ad617806b1.png)

