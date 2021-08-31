import numpy as np
import h5py

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import seaborn as sns
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model



trainFile = h5py.File('train_happy.h5', "r")
testFile = h5py.File('test_happy.h5', "r")



train_x = np.array(trainFile['train_set_x'][:])
train_y = np.array(trainFile['train_set_y'][:])

test_x = np.array(testFile['test_set_x'][:])
test_y = np.array(testFile['test_set_y'][:])
print(train_x.shape)
print(train_y.shape)



train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

print(train_y.shape)
print(test_y.shape)



# plt.imshow(train_x[0])



X_train = train_x / 255.0
X_test = test_x / 255.0

y_train = train_y.T
y_test = test_y.T



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



epochs = 30
batch_size = 30

# from ann_visualizer.visualize import ann_viz
#
# ann_viz(model,view=True ,title="My first neural network")

history = model.fit(x=X_train, y=y_train, epochs=epochs, verbose=2,batch_size=batch_size)


test_score = model.evaluate(X_test, y_test, verbose=1)

Y_pred = model.predict_classes(X_test)

print('test loss:', test_score[0])
print('test accuracy:', test_score[1])

print(history.history.keys())

from keras.preprocessing import image
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import preprocess_input

img_path = 'bibiSad.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(model.predict(x))

img_path = 'bibiSmile.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(model.predict(x))

training_accuracy = history.history['accuracy']
training_loss = history.history['loss']

# E = range(len(training_accuracy))
# plt.plot(E, training_accuracy, color='red', label='Training accuracy')
# plt.title('epochs vs Training accuracy')
# plt.legend()
#
# plt.figure()
# plt.plot(E, training_loss, color='red', label='Training Loss')
# plt.title('epochs vs Training Loss')
# plt.legend()

# plt.show()

cm = confusion_matrix(y_test,Y_pred)
sns.heatmap(cm,annot=True)
plt.show()

plot_model(model, to_file='HappyModel.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))