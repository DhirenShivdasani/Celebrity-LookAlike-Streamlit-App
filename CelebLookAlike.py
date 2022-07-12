from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import numpy as np
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from matplotlib.pyplot import *
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox






def get_cropped_face(file):
    img = cv.imread(file)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar = cv.CascadeClassifier('haar_face.xml')

    faces = haar.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        if len(faces) == 1:
            return roi_color

""" loads image """
def load_img(path):
    img = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
    return img


#load mtcnn and InceptionResnetV1
def load_models():
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet.eval()
    mtcnn = MTCNN()
    mtcnn.eval()
    return resnet, mtcnn

resnet, mtcnn = load_models()

"""finds face in the orginal image and returns only the face""" 
def img_to_mtcnn(img):
    img_cropped = mtcnn(img) #detects face in image
    return img_cropped

"""loads mtcnn"""
def load_mtcnn(mtcnn):
    return mtcnn.permute(1,2,0).detach().numpy()

"""returns the image embedding from the mtcnn image"""
def mtcnn_to_embedding(mtcnn):
    img_embedding = resnet(mtcnn.unsqueeze(0)) #gets emebdding
    return img_embedding.detach().numpy()

#shows the image embedding 
def show_embed(img_embedding):
    embed_img = np.reshape(img_embedding, (32,16)) 
    embed_img = embed_img / embed_img.max()
    embed_img_int = (embed_img * 255).astype(np.uint8)
    return embed_img_int

#returns the image embedding just from inputting the path of the image
def path_to_embedding(path):
    img = load_img(path)
    mtcnn_test = img_to_mtcnn(img)
    img_embed = mtcnn_to_embedding(mtcnn_test)
    return img_embed

#load in text file with all celebrity names 
def make_celeb_list():
    fullCelebList = []
    with open('celebNames.txt', 'r') as f:
        for line in f:
            fullCelebList.append(line.strip('\n'))
    return fullCelebList


#function for calculating euclidean distance
def euclidean_distance(p1,p2):
    return np.linalg.norm(p1-p2)

def calculate_distances(path, img_embedding_df): #takes a path for any image
    img_embed = path_to_embedding(path) #gets the image embedding from the path
    distances = [euclidean_distance(img_embed ,row.values)
     for i, row in img_embedding_df.drop(['name'], axis = 1).iterrows()] #calculates the euclidean distance between the image embedding inputted and the rest of the celebrity image embeddings
    return distances #return a list of the distances


#makes data frame of all the distances, sorted from lowest to highest
def make_distances_df(uploaded_img_path, img_embedding_df):
    df = pd.DataFrame( data = 
                      {'celeb': img_embedding_df['name'],
                       'distance': calculate_distances(uploaded_img_path, img_embedding_df)}) 
    df.sort_values('distance',ascending = True, inplace = True) 
    return df


def make_barplot(df):
    fig = plt.figure(figsize = (12, 6))

    sns.barplot(x = df['celeb'][:50], y = df['distance'][:50], data = df, ci = None)
    plt.xticks(rotation = 90)
    plt.title('Euclidean distance from the uploaded image to each of the celebrity images', fontsize = 20)

    return fig


"""gets the exact image of the celebrity that looks closest alike to the uploaded image"""
def get_celeb_look_alike_img(df, img_embedding_df):
    celeb_df = img_embedding_df[img_embedding_df['name'] == df.iloc[0]['celeb']] #subset dataframe for the celeb look alike
    

    for i in range(len(celeb_df)): 
        if celeb_df.index[i] == df.index[0]: #finds the index of the image embedding that matches the celebrity look alike
            celeb_img = load_img(f"CelebImages/Cropped/{df.iloc[0]['celeb']}Cropped/{df.iloc[0]['celeb']}Cropped{i+1}.jpg") #gets the exact image of the celebrity that looks closest alike to the uploaded image
    return celeb_img


def add_uploaded_embed(img_embed_df, embed, name):
    embed=np.append(embed, 0) #Adding a 0 to make it the same size as the dataframe shape (will change to name l8r)
    next_row = len(img_embed_df) #Next row to work with
    img_embed_df.loc[next_row] = embed #Append the embed to a new row
    img_embed_df.loc[next_row, "name"] = name #Change the 0 back to the right name
    return img_embed_df


def make_tsne(img_embed_df):
    tsne = TSNE(n_components = 2, learning_rate = 200, random_state = 0)  
    tsne_features = tsne.fit_transform(img_embed_df.drop(['name'], axis = 1))  #makes T-SNE
    xs = tsne_features[:,0]
    ys = tsne_features[:,1]
    
    tsne_df = pd.DataFrame(data={"name":img_embed_df['name'], 'xs':xs, 'ys':ys}) #data frame to store all the x and y's for each celeb 
    return tsne_df


def get_top3_xs(tsne_df, top3):  
    top3_xs = []
    for celeb in top3:
        tsne_df_top3 = tsne_df[tsne_df['name'] == celeb]
        top3_xs.append(tsne_df_top3['xs'][tsne_df_top3.index[0]])  #gets the t-sne x values for the closest 5 celebrities 
        
    return top3_xs

def get_top3_ys(tsne_df, top3):  
    top3_ys = []
    for celeb in top3:
        tsne_df_top3 = tsne_df[tsne_df['name'] == celeb]
        top3_ys.append(tsne_df_top3['ys'][tsne_df_top3.index[0]])  #gets the t-sne x values for the closest 5 celebrities 
        
    return top3_ys


def make_plot(path, img_embed_df):
    
    #gets embedding for the uploaded image
    uploaded_img_embed = path_to_embedding(path)
    
    #makes data frame for all the euclidean distance values from the celebrity image embedding to the uploaded image embedding
    distances_df = make_distances_df(path, img_embed_df)

    #adds the uploaded image embedding to the embedding data frame
    final_embed_df = add_uploaded_embed(img_embed_df, uploaded_img_embed, 'dhiren') #adds uploaded img to data frame    
    
    #tsne data frame for all the image embeddings
    tsne_df = make_tsne(final_embed_df)
    
    #top 5 closest celebrities to the uploaded image
    top3 = distances_df['celeb'].unique()[0:3]
    
    #gets the images of the top 5 celebrities closest to the uploaded image
    top3_imgs = [cv2.cvtColor(cv2.imread(f"CelebImages/Cropped/{fn.split(' ')[0]} {fn.split(' ')[1]}Cropped/{fn.split(' ')[0]} {fn.split(' ')[1]}Cropped1.jpg"),cv2.COLOR_BGR2RGB) for fn in top3]
    
    #create scatterplot
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    sns.scatterplot(x='xs', y='ys', hue='name', data=tsne_df)
    
    #top 5 x's and y's for the 5 closest celebrities 
    top3_xs = get_top3_xs(tsne_df, top3)
    top3_ys = get_top3_ys(tsne_df, top3)
    
    box_coords = [(-200,0), (50,100), (-100,60)]
    #plots the images of these celebrites over their point values 
    for x, y, img, coord in zip(top3_xs, top3_ys, top3_imgs, box_coords):
        im = OffsetImage(img, zoom = .7)
        ab = AnnotationBbox(im, (x,y), xybox = coord, xycoords = 'data', boxcoords = 'offset points', pad = .3, arrowprops = dict(arrowstyle = '->') )

        ax.add_artist(ab)
        ab.set_visible(True)
        
    
    #for annotating the uploaded image
    self_x =tsne_df['xs'].values[-1]
    self_y =tsne_df['ys'].values[-1]
    
    ax.annotate("You!",
            xy=(self_x,self_y), xycoords='data',
            xytext=(self_x+1, self_y-1), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
    
    
    ax.get_legend().remove()

    #drops the uploaded image from the data frame to return the data frame back to normal 
    final_embed_df.drop(856, axis = 0, inplace = True)
    return fig





#used to add celebrities to the sample
def addCelebrity(first, last):
    firstName = first
    lastName = last
    getImages(firstName, lastName)
    count = 1
    os.mkdir(f'CelebImages/Cropped/{firstName} {lastName}Cropped')
    for entry in os.scandir(f'CelebImages/{firstName} {lastName}'):
        roi_color = get_cropped_face(entry.path)
        if roi_color is not None:
            cv2.imwrite(f'CelebImages/Cropped/{firstName} {lastName}Cropped/{firstName} {lastName}Cropped{count}.jpg', roi_color)
            count +=1



#used for getting a list of 500 celebrities off the internet
def find_celeb_names():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options = options, executable_path = '/Users/dhirenshivdasani/Downloads/chromedriver')
    driver.get('https://www.famousfix.com/list/500-actors-actresses')
    driver.implicitly_wait(3)

    fullCelebList = []
    y = 100
    for i in range(1,200):
        driver.execute_script("window.scrollTo(0,"+ str(y) + ")")
        driver.implicitly_wait(16)
        fullCelebList.append(driver.find_element(By.XPATH, f'//*[@id="container"]/li[{i}]/a/span/span').text)
        y += 100

    return fullCelebList



if __name__ == '__main__':
    resnet, mtcnn = load_models()
    all_embeds = pd.read_csv('img_embeddings134.csv', index_col = [0])




