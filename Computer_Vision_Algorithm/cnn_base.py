import numpy as np
import time

from scipy import signal

# tf imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#
# r_arr = np.array([
#     [46,  67,  161, 250, 223, 169],
#     [48,  41,  114, 65,  159, 104],
#     [101, 165, 216, 231, 230, 196],
#     [54,  255, 145, 87,  106, 111],
#     [233, 138, 206, 233, 134, 145],
#     [148, 255, 75,  43,  139, 176]
# ])

r_test = np.random.randint(0, 255, size=(400, 400))  # 800, 600

r_arr = r_test

# g_arr =  np.array([
#     [48,  55,  120, 175, 113, 40],
#     [33,  20,  80,  19,  96,  36],
#     [53,  126, 195, 234, 255, 248],
#     [0,   238, 118, 105, 185, 219],
#     [151, 67,  157, 230, 202, 250],
#     [60,  191, 13,  29,  203, 255]
# ])
#
#
# b_arr =  np.array([
#     [60, 55,  102,  152, 100, 34],
#     [40, 15,  53,   0,   78,  23],
#     [49, 109, 152,  187, 235, 225],
#     [0,  201, 75,   55,  128, 159],
#     [69, 1,   125,  187, 103, 121],
#     [0,  112, 0,    0,   81,  116]
# ])

kernel = np.array(  [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# kernal = np.array(  [
#     [-1, 0, 1, 3, 2, 1, 3],
#     [-2, 0, 2, 3, 2, 1, 3],
#     [-1, 0, 1, 3, 2, 1, 3],
#     [-1, 0, 1, 3, 2, 1, 3],
#     [-1, 0, 1, 3, 2, 1, 3],
#     [-2, 0, 2, 3, 2, 1, 3],
#     [-1, 0, 1, 3, 2, 1, 3]
# ])

if kernel.shape[0] % 2 == 0:
    padding_r = int((kernel.shape[0] - 2) / 2)
else:
    padding_r = int((kernel.shape[0] - 1) / 2)

if kernel.shape[0] % 2 == 0:
    padding_c = int((kernel.shape[1] - 2) / 2)
else:
    padding_c = int((kernel.shape[1] - 1) / 2)

h_in, w_in = r_arr.shape[0], r_arr.shape[1]
h_k, w_k = kernel.shape[0], kernel.shape[1]

kernel_stride = 1
result = np.zeros((h_in, w_in)).astype(int)
# result = []
start = time.time()

for h in range(0, h_in, kernel_stride):
    for w in range(0, w_in, kernel_stride):
        patch = r_arr[h:h + h_k, w:w + w_k]
        if patch.shape == kernel.shape:
            result[h, w] += np.sum(kernel * patch)

end = time.time()
print("own conv:", end - start)
print(np.array(result))

start = time.time()
test_res = signal.correlate2d(r_arr, kernel, "valid")
end = time.time()
print("signal conv:", end - start)
print(test_res)


def vgg16_self():
    # vgg16
    import keras, os
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np

    trdata = ImageDataGenerator()
    traindata = trdata.flow_from_directory(directory="data", target_size=(224, 224))
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory="test", target_size=(224, 224))

    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    from keras.optimizers import Adam
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    from keras.callbacks import ModelCheckpoint, EarlyStopping
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
    hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata, validation_steps=10,
                               epochs=100, callbacks=[checkpoint, early])

    import matplotlib.pyplot as plt
    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()


    # TEST
    from keras.preprocessing import image
    img = image.load_img("image.jpeg", target_size=(224, 224))
    img = np.asarray(img)
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    from keras.models import load_model
    saved_model = load_model("vgg16_1.h5")
    output = saved_model.predict(img)
    if output[0][0] > output[0][1]:
        print("cat")
    else:
        print('dog')

def tf_function():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()


# print("TF TEST")
# tf_function()
