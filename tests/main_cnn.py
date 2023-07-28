import sys
sys.path.insert(0, "../gann")

# pip3 install cifar10
# conda install scipy 

import cifar10
import numpy as np
import matplotlib.pyplot as plt

imgs = []
lbls = []
qty = 10
for image, label in cifar10.data_batch_generator():
    imgs.append(image) # numpy array of an image, which is of shape 32 x 32 x 3
    lbls.append(label) # integer value of the image label
    if len(imgs) >= qty:
        break

imgs = np.array(imgs)
lbls = np.array(lbls)
print(imgs.shape)
#print(lbls.shape)
# Class names in CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# create a figure
plt.figure(figsize=(10, 5))
# Show 10 images
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(imgs[i])
    plt.title(class_names[lbls[i]])
    plt.axis('off')

# Show the plot
#plt.show()

# Normalize the images
imgs = imgs / 255.0

# One-hot encode the labels
from gann.utils import to_categorical
lbls = to_categorical(lbls)

# Split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imgs, lbls, test_size=0.2, random_state=42)

# Build the model
from gann.utils import to_categorical
from gann.nn import Net
from gann.nn.layers import Dense, Flatten, Conv2D #, MaxPooling2D, Dropout
from gann.nn.listeners import Listener,Keep_Progress, TwoDimensional_Graph
from gann.nn.activations import Activation, ReLU, Softmax
from gann.nn.losses import Loss, CategoricalCrossEntropy

print('Activations availables:', Activation.all())
print('Losses availables:', Loss.all())
print('Listeners availables:', Listener.all())

model = Net(loss_func='CategoricalCrossEntropy', listener=Keep_Progress())
model.add(Conv2D(8, (3, 3), padding=0, activation='ReLU'))
model.add(Conv2D(8, (3, 3), padding=0, activation='ReLU'))
model.add(Conv2D(16, (3, 3), padding=0, activation='ReLU'))
model.add(Flatten())
model.add(Dense(2, activation='ReLU'))
model.add(Dense(10, activation='Softmax'))
model.compile((32, 32, 3))
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, lr=0.01) #, batch_size=32)

