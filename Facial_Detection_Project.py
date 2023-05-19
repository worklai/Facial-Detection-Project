# install the library we need
!pip install opencv-python
from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pdb

#import the name and category
train_name = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ece595ml-s2022/train.csv")  # local drive how to direct get the data
category = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ece595ml-s2022/category.csv")
train_name = train_name.merge(category, left_on='Category', right_on='Category', how='left')

training_data_b = [] # big training data
error_img_b = []
t=[]
def create_training_data():
        path =  "/content/drive/MyDrive/train/"
        for i in tqdm(range(69539)): 
            img = str(i)+str(".jpg" )           
            try:
              
              img_array = cv2.imread(path + img ,cv2.IMREAD_COLOR)         
              new_array = cv2.resize(img_array,(100,100))
              training_data_b.append(new_array)
              # t.append(new_array)
            except Exception as e:  
              error_img_b.append(img)     
create_training_data()
np.shape(training_data)
np.shape(training_data_b)
x_train = np.stack(training_data,0)
np.shape(x_train)
yy_train = train_name[~train_name['File Name'].isin(error_img_b)]
y_train = list(yy_train.iloc[:13772,3].astype("int32"))


num_classes = 100
input_shape = (100, 100, 3)
y_train = keras.utils.to_categorical(y_train, num_classes)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(4, 4), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)


model.summary()

#run model
batch_size = 128
epochs = 100
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, validation_split=0.1 )
#Show the result
score = model.evaluate(x_train, y_train, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

x_test=[]
np.shape(org_tr_data)
x_test.append(org_tr_data[1])
np.shape(x_test)
x_test = []  #no crop


def testing_data():
        path =  "/content/drive/MyDrive/test/test/"
        for i in tqdm(range(4977)): 
            img = str(i)+str(".jpg" )
            try:
              img_array = cv2.imread(path+img,cv2.IMREAD_COLOR) 
              new_array = cv2.resize(img_array,(100,100))  
              x_test.append(new_array)
            except Exception as e:
              #just add a name attribute  
              x_test.append(org_tr_data[65])
testing_data()
x_test = np.stack(x_test,0)
x_test.shape
category = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ece595ml-s2022/category.csv',skiprows=[0], names = ['Num','Category'])
y_test =[]
id = []
for i in tqdm(range(4977)):
  yhat = model.predict(x_test[i:i+1])
  index = np.where(yhat ==yhat.max())[1][0]
  y_test.append(category.loc[category['Num'] ==index, 'Category'].tolist()[0])


Output = {'Category': y_test}


Output_df = pd.DataFrame(Output) 


Output_df.to_csv('y_test_big.csv', index=True)  

