from glob import glob

import cv2 as cv
import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras import losses

IMM_SIZE = 224
fitting_save = True


def get_data(folder):
    data = []
    images = glob(folder + "/*")
    class_name = folder.rsplit("/", 1)[-1]

    for img in images:
        image = cv.imread(img)
        if image is not None:
            image = cv.resize(image, (IMM_SIZE, IMM_SIZE))
        if image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        data.append([image, class_name])

    return data


folder_military = "dataset/test/millitary"
folder_other = "dataset/test/other"
test_military = get_data(folder_military)
test_other = get_data(folder_other)
test = test_military + test_other

folder_military = "dataset/train/millitary"
folder_other = "dataset/train/other"
train_millitary = get_data(folder_military)
train_other = get_data(folder_other)
train = train_millitary + train_other


x_train = []
y_train = []
x_test = []
y_test = []

for i in range(0, len(train)):
    x_train.append(train[i][0])  # image
    y_train.append(train[i][1])  # label
for i in range(0, len(test)):
    x_test.append(test[i][0])
    y_test.append(test[i][1])


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)


# normalizing images
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

# reshaping input images
x_train = x_train.reshape(-1, IMM_SIZE, IMM_SIZE, 1)
x_test = x_test.reshape(-1, IMM_SIZE, IMM_SIZE, 1)


y_train = np.array(y_train)
y_test = np.array(y_test)


# base model
from keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMM_SIZE, IMM_SIZE, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

for layer in base_model.layers:
    layer.trainable = False
base_model.summary()


# our model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), shuffle=True)

# save model
import pickle

if fitting_save:
    model_json = model.to_json()
    with open("model.json", "w") as file:
        file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

    # with open('history.pickle', 'wb') as f:
    #     pickle.dump(history.history, f)
    # with open('lab.pickle', 'wb') as f:
    #     pickle.dump(lab, f)

for img in x_test:
    prediction = model.predict(img)
    print(prediction)


def vehicleType(img_path):
    img = cv.imread(img_path)

    if img is not None:
        img = cv.resize(img, (IMM_SIZE, IMM_SIZE))
    if img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    pred = model.predict_classes(img)
    return pred


img = "dataset/test/other/other_0_1006.jpeg"
classOfImage = vehicleType(img)

print(f"{img} class is {classOfImage}")


# accuracy
z = model.predict_classes(x_train) == y_train
scores_train = sum(z + 0) / len(z)
z = model.predict_classes(x_test) == y_test
scores_test = sum(z + 0) / len(z)

print('Train DataSet accuracy: {: .1%}'.format(scores_train), 'Test DataSet accuracy: {: .1%}'.format(scores_test))
