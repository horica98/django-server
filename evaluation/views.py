# import keras
import tensorflow
from django.shortcuts import render
# Create your views here.
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http import JsonResponse
import numpy as np
from keras.preprocessing import image
import cv2
from sklearn import preprocessing
# from django.conf import settings
from rest_framework.utils import json
import os
from os import listdir
from os.path import isfile, join
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def preprocess_images(source_image, symbols_path, extension, symbols_bb_arr, index):
    image = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    image = 255 - image
    for symbol_bb in symbols_bb_arr:
        y1, y2, y3, y4, x1, x2, x3, x4 = symbol_bb
        xmin, xmax, ymin, ymax = min(x1, x2, x3, x4), max(x1, x2, x3, x4), min(y1, y2, y3, y4), max(y1, y2, y3, y4)
        height, width = xmax - xmin, ymax - ymin

        image_cropped_0 = image[xmin:xmin + height, ymin:ymin + width]
        image_cropped_0_3 = image[xmin:xmin + height, ymin - 3:ymin + width + 3]
        image_cropped_0_7 = image[xmin:xmin + height, ymin - 7:ymin + width + 7]

        destination_path_0 = symbols_path + str(index) + '_0' + extension
        destination_path_0_3 = symbols_path + str(index) + '_0_3' + extension
        destination_path_0_7 = symbols_path + str(index) + '_0_7' + extension

        cv2.imwrite(destination_path_0, image_cropped_0)
        cv2.imwrite(destination_path_0_3, image_cropped_0_3)
        cv2.imwrite(destination_path_0_7, image_cropped_0_7)
        index += 1


def edit_created_images(source, destination):
    for file_name in listdir(source):
        image = cv2.imread(source + "\\\\" + file_name, 0)
        image = cv2.resize(image, (80, 128))
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)

        image_3_1 = cv2.erode(image, kernel, iterations=1)
        image_3_2 = cv2.erode(image, kernel, iterations=2)
        kernel = np.ones((5, 5), np.uint8)
        image_5_1 = cv2.erode(image, kernel, iterations=1)
        image_5_2 = cv2.erode(image, kernel, iterations=2)

        index = file_name.rfind('.')
        name = file_name[:index]
        cv2.imwrite(destination + "\\\\" + name + '.jpeg', image)
        cv2.imwrite(destination + "\\\\" + name + '_3_1.jpeg', image_3_1)
        cv2.imwrite(destination + "\\\\" + name + '_3_2.jpeg', image_3_2)
        cv2.imwrite(destination + "\\\\" + name + '_5_1.jpeg', image_5_1)
        cv2.imwrite(destination + "\\\\" + name + '_5_2.jpeg', image_5_2)

    return


def update_folder_name():
    with open(r'D:\licenta_horea\django_app\current_directory_name') as file:
        lines = file.readlines()
        path = lines[0].strip()
        index = path.rfind('_')
        new_path = path[:index + 1] + str(int(path[index + 1:]) + 1)
    with open(r'D:\licenta_horea\django_app\current_directory_name', 'w') as file:
        file.write(new_path)
    os.mkdir(path)
    return path


loaded_model = tensorflow.keras.models.load_model(r'D:\model\model_operands_20_validation.h5')
print(loaded_model.summary())

# class Predictor(View):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
@api_view(['POST'])
@csrf_exempt
def proba(request):
    a = request
    data = request.data
    filename = data['filename']
    arrays = data['array']

    folder = update_folder_name()
    preprocess_images(filename, folder + '\\\\' + 'symbol_', '.jpeg', arrays, 0)
    edit_created_images(folder, folder)

    le = preprocessing.LabelEncoder()
    character_curated = [ord(c) for c in '!%&()*+-./:;<=>[]{|}']
    ids = le.fit_transform(character_curated)

    batch = np.empty((0, 128, 80, 1))
    i = 0

    file_names = listdir(folder)
    file_names = sorted_alphanumeric(file_names)

    print(loaded_model.summary())
    for file_name in file_names:
        print(file_name)
        img = image.load_img(folder + "\\\\" + file_name, color_mode="grayscale")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        batch = np.concatenate((batch, x))
    classes = loaded_model.predict(batch, batch_size=32)
    np.set_printoptions(threshold=np.inf)

    classes = np.vsplit(classes, classes.shape[0] / 15)
    classes_means = np.mean(classes, axis=1)
    classes = np.array(np.argmax(classes_means, axis=1))
    # return classes, classes_means
    print(classes)
    print(classes_means)
    a = [1, 2, 3]
    # response = JsonResponse(a)
    return JsonResponse(data={'classes': classes.tolist(), 'means': classes_means.tolist()}, safe=False)
