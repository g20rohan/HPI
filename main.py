from __future__ import print_function

from flask import Flask, render_template, request
import tensorflow as tf
import os
from io import open
import shutil
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
from keras.models import Model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import json
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def resnet_module(input, channel_depth, strided_pool=False):
    residual_input = input
    stride = 1

    if (strided_pool):
        stride = 2
        residual_input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same",
                                kernel_initializer="he_normal")(residual_input)
        residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal")(
        input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_first_block_first_module(input, channel_depth):
    residual_input = input
    stride = 1

    residual_input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(
        residual_input)
    residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_block(input, channel_depth, num_layers, strided_pool_first=False):
    for i in range(num_layers):
        pool = False
        if (i == 0 and strided_pool_first):
            pool = True
        input = resnet_module(input, channel_depth, strided_pool=pool)

    return input


def ResNet50(input_shape, num_classes=10):
    input_object = Input(shape=input_shape)
    layers = [3, 4, 6, 3]
    channel_depths = [256, 512, 1024, 2048]

    output = Conv2D(64, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(input_object)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(output)
    output = resnet_first_block_first_module(output, channel_depths[0])

    for i in range(4):
        channel_depth = channel_depths[i]
        num_layers = layers[i]

        strided_pool_first = True
        if (i == 0):
            strided_pool_first = False
            num_layers = num_layers - 1
        output = resnet_block(output, channel_depth=channel_depth, num_layers=num_layers,
                              strided_pool_first=strided_pool_first)

    output = GlobalAvgPool2D()(output)
    output = Dense(num_classes)(output)
    output = Activation("softmax")(output)

    model = Model(inputs=input_object, outputs=output)

    return model
# ----------------- The Section Responsible for Inference ---------------------
CLASS_INDEX = None
execution_path = os.getcwd()

MODEL_PATH = os.path.join(execution_path, "C:/Users/g10ro/Documents/Projects/Prof/trained_models/RN50_Raw_Data.h5")
JSON_PATH = os.path.join(execution_path, "C:/Users/g10ro/Documents/Projects/Prof/trained_models/model_class.json")


def preprocess_input(x):
    x *= (1. / 255)

    return x


def decode_predictions(preds, top=5, model_json=""):
    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)

    return results


def run_inference():
    model = ResNet50(input_shape=(224, 224, 3), num_classes=10)
    model.load_weights(MODEL_PATH)

    picture = os.path.join(execution_path, "/content/drive/MyDrive/Proff/test/chef/chef-10.jpg")

    image_to_predict = image.load_img(picture, target_size=(
        224, 224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)

    image_to_predict = preprocess_input(image_to_predict)

    prediction = model.predict(x=image_to_predict, steps=1)

    predictiondata = decode_predictions(prediction, top=int(5), model_json=JSON_PATH)

    for result in predictiondata:
        print(str(result[0]), " : ", str(result[1] * 100))
def getvalue():
    shutil.rmtree('C:/Users/g10ro/Documents/Projects/Prof/static')
    directory = "static"
    parent_dir = "C:/Users/g10ro/Documents/Projects/Prof/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    original_image=request.files.getlist("image")
    for upload in request.files.getlist("image"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
    destination = "/".join([path, filename])
    print("Accept incoming file:", filename)
    print("Save it to:", destination)
    upload.save(destination)
    activation = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    #model = load_model('C:/Users/g10ro/Documents/Projects/Alzheimer/trained_models/Alzheimer_Stage_detection_lrelu.h5',custom_objects={'<lambda>': activation})
    model = ResNet50(input_shape=(224, 224, 3), num_classes=10)
    model.load_weights(MODEL_PATH)
    image_to_predict = image.load_img(destination, target_size=(
        224, 224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    image_to_predict = preprocess_input(image_to_predict)
    prediction = model.predict(x=image_to_predict, steps=1)
    predictiondata = decode_predictions(prediction, top=int(5), model_json=JSON_PATH)
    for result in predictiondata:
        label=str(result[0]), " : ", str(result[1] * 100)

    #my_new_list = [i * 10 for i in prediction]
    #mild_precent=my_new_list[0]
    #nondem=my_new_list[1]
    #verymild=my_new_list[2]
    #print('prediction class:', predictions)



    #os.remove(destination)
    return render_template('pass.html', img=filename, labelname=label)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
