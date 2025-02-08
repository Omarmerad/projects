# Simple image classifier 

in here, we are going to implement a very basic image classifier using [[CNN]] using python, we are going to create two files, the first one called image_classifier to train the model, and then test.py to test it with our own examples, lets start with image_classifier.py  :

###### - Importing necessary libraries :

``` 
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models 
```

 ###### - Loading the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) images :

```
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# to trap the values between 0 and 1

training_images, testing_images = training_images / 255, testing_images / 255
```

###### - Verifying that the dataset is correctly loaded :

```
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'truck']

  

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.xticks([])

    plt.xticks([])

    plt.imshow(training_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[training_labels[i][0]])

  

plt.show()
```

###### - Creating the convolutional base :

```
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
```

>  Explanation :
layers.Conv2D(...):  it applies a set of filters to the input data, to detect patterns, 32 stands for the number of filters, 3x3 is the filter size (we are going to go by 3x3 grids ), and the [relu](https://builtin.com/machine-learning/relu-activation-function) activation function is used to only keep the positive values.
>as we go into deeper layers, the number of filters increases because we are going to detect shapes and objects, and there is no need for an input shape because the network automatically knows the input of the previous layer.
>as for [[MaxPooling2D]], it halves the map size and takes only the most significant features.
>Flatten() just takes the multidimensional data and gathers it in a 1D array to simplify the data before the fully connected layers (they usually make the final prediction, which in our case is the [[Dense layer]])


###### - Training the model and saving it :
```
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit (training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss : {loss}")
print(f"accuracy: {accuracy}")

model.save('image_classifer.keras')
```


## testing with our own examples :

```
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

```

In here we load the model first, and then we import the image we want to test, we resize it to fit our model (32x32 pixels) and then we flip the color module to fit again, and finally we print our prediction.


