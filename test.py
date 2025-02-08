import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifer.keras')

img = cv.imread('C:/Users/hp/Desktop/ai project/testing_image.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")
img = cv.resize(img, (32, 32))
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Predicition is {class_names[index]}')
