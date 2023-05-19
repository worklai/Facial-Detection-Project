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
train_name
