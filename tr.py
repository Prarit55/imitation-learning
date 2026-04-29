import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,GlobalAveragePooling2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import cv2

train_data = np.load('data/tf_data.npy', allow_pickle=True)

noise = 0.05
images = []
controls = []
i = 0
forward = 0
backward = 0
normal = 0
print(len(train_data))


for data in train_data:
    i = i+1
    source_image = data[0]
    new_image = cv2.resize(source_image, (200, 66))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2HLS)
    new_image = cv2.GaussianBlur(new_image, (3, 3), 0)
    new_image = new_image/255 - 0.1
    control = data[1]

    # Data cleaning
    if control[1] > 0.2:
        forward += 1
        images.append(new_image)
        controls.append(control)

    elif -0.3 > control[0] > 0.3 or control[1] < 0.1:  # The images with negative throttle and high steering angle are duplicated
        backward += 2
        copy_image = new_image - 0.05
        if control[0] < 0:
            copy_control = [control[0] - noise, control[1] - 0.1]
        else:
            copy_control = [control[0] + noise, control[1] - 0.1]

        images.append(new_image)
        controls.append(control)
        images.append(copy_image)
        controls.append(copy_control)
    else:
        if i % 3 == 0:
            normal += 1
            images.append(new_image)
            controls.append(control)

print("total images:", len(images))
print("forward filter", forward)
print("backward filter", backward)
print("normal images", normal)

X_train = np.array(images, dtype='float16')
y_train = np.array(controls)

X_train, y_train = shuffle(X_train, y_train)

print("the image is being trained on {} samples".format(len(X_train)))
epochs = 10
batch_size = 128
# Load VGG19
vgg_model = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(66, 200, 3)
)

# Freeze most layers
for layer in vgg_model.layers[:-4]:
    layer.trainable = False

# Build Sequential model
model = Sequential([
    vgg_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='linear')
])

model.compile(optimizer='Adam', loss='mse')
print('training...')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, verbose=2, batch_size=batch_size)

print()
print('training complete')

model.save('models/tf_model_vggup.h5')
print()
print('model saved')