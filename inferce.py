import tensorflow as tf
import cv2
import numpy as np
import matplotlib as plt
from PIL import Image

path='val/C1_thinF_IMG_20150604_104722_cell_15.png'


model = tf.keras.models.load_model('Malaria')

data=Image.open(path)
new_image = data.resize((50, 50))
new_image1 = np.array(new_image).reshape(-1,50,50,3)

if ((model.predict([new_image1]))==[0]):
  print('Parasitized')
  data.show()
  
else:
  print('Uninfected')
  data.show()