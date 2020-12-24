#!/usr/bin/python3
from flask import Flask, json,jsonify
from flask import request
import numpy as np
import base64
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import os
import cgi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to suppress all the warning messages


def predict(img_data, model):
    img_data = str.encode(img_data.split(',')[1])
    with open("tempDigit.jpg", "wb") as fh:
        fh.write(base64.decodebytes(img_data))

    im = Image.open('tempDigit.jpg')
    im = im.resize((28,28))
    im.save('tempDigit.jpg')

    im = cv2.imread('tempDigit.jpg',0) 
    #plt.imshow(im.reshape(28,28))
    
    threshold = 64 # to clear the noise
    im = im.reshape(28,28) 
    
    for r in range(len(im)):
        for c in range(len(im[r])):
            im[r][c] = 0 if im[r][c] <= threshold else 1

    im = im.reshape(1,28,28,1)
    
    res = list(model.predict_classes(im))[0]

    return res




api = Flask(__name__)
model = load_model('digit_recognizer.h5')

@api.route('/digit_recognizer', methods=['GET', 'POST'])
def predict_digit():
    data = request.get_json()
    img_data = str(data["img_data"])
    return str(predict(img_data, model)) #the return value must be a string...

if __name__ == '__main__':
    api.run(host='0.0.0.0')





























