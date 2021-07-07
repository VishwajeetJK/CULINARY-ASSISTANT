# CULINARY-ASSISTANT

● The First Objective is the Image Recognition of dishes using Deep Learning. We will
be using Convolutional Neural Networks to detect the dish present in the image so
that the user can upload the photograph of the dish and the model will return the name
of the dish.

<p align="center">
    <image src="screenshots/1.png" width="650">
</p>

● The second objective on which we will be focussing is the recommendation system
which will recommend complimentary dishes to the user related to the dish in the
photograph which the user has uploaded. This will help the user get insights about the
other side dishes which are often taken with that main dish.
      
 <p align="center">
    <image src="screenshots/2.png" width="650">
</p>

● We will also be creating Rest APIs to host our models on the web app and we will
also be identifying the APIs which will give us the recipe of the dishes. Finally, will
make the front end for the web app which should be user friendly and very intuitive in
nature. It shall attempt to request the APIs with an image of the dish and display the
recipe and recommendations back on itself.
      
 <p align="center">
    <image src="screenshots/3.png" width="650">
</p>

Data Acquisition: 
A dataset is used to train the CNN model consisting of various
categories. The custom dataset which was made considered the data from FOOD 101 dataset
which consists of 101 classes and each class consists of 1,000 images. So a total of 1,01,000
images are available. This is a great thing as for deep learning architectures to generalize well
we need a large dataset and here we already have it. But this has it's pros and cons which is
that it requires a lot of computing power and our laptops computing power cannot match
those standards. So we have used Kaggle.com to utilize the GPU available there. If we face
any problem while training on this huge dataset we have a couple of datasets which are
smaller than this, which can be used to train our model.
     
<p align="center">
    <image src="screenshots/4.png" width="700">
</p>

Data Modelling: 
Object recognition or cooking court recognition using Convolutional
Neural Networks and the search for the nearest neighbors in a record of all the images. This
combination helps to find the correct recipe more likely, as the top-5 categories of the CNN
are compared to the next-neighbor category with ranked correlation.
    
<p align="center">
    <image src="screenshots/5.1.png" width="450">
    <image src="screenshots/5.2.png" width="450">
</p>

Datasets Referred:
The most widely used dataset is the Food Images (Food-101) dataset, which is a subset of the
actual dataset uploaded by K Scott Mader on Kaggle. This dataset contains sets of images of
different dishes from around the world. There are approximately 101,000 images in total. The
subset of the dataset that we have used is divided into the following 20 categories with each
category having 1000 images.
['Cheesecake','Chicken_Curry','Chicken_Wings','Chocolate_Cake','Chocolate_Mousse','Cup_
Cakes','French_Fries','French_Toast','Fried_Rice','Garlic_Bread','Ice_Cream','Macaroni_And
_Cheese','Nachos','Omelette','Pancakes',’'Pizza','Samosa','Spring_Rolls','Strawberry_Shortcak
e','Waffles']

[FOOD RECOMMENDATION SYSTEM DATASET](https://docs.google.com/spreadsheets/d/1DP_M59KzT8rOp1Fe6OSbwNijgCmgcKq5Rbe9gk11znw/edit?usp=sharing)
        
# 1st Example:
<p align="center">
    <image src="screenshots/6.png" width="450">
    <image src="screenshots/7.png" width="450">
</p>
## After clicking on the predict button:
   
<p align="center">
    <image src="screenshots/8.png" width="450">
    <image src="screenshots/9.png" width="450">
</p>
