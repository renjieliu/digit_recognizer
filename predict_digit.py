#!/usr/bin/python3

import cgi
import os

form = cgi.FieldStorage()
img_data = form.getvalue('img_data')
command = 'curl -XPOST -H "Content-Type: application/json" --data \'{"img_data":"' + img_data + '"}\' http://192.168.1.206:5000/digit_recognizer'
res = os.popen(command).read()
print("Content-Type: text/html")    # HTML is following
print()                             # blank line, end of headers
print("<html><head></head><body>")
print(res)#this is the value to be extracted for showing on the webpage
print("</body></html>")








#import numpy as np
#import base64
#from PIL import Image
#import cv2
##import tensorflow as tf
#from tensorflow.keras.models import load_model
#import os
#import cgi
#
#
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to suppress all the warning messages
#
#
#def predict(img_data):
#	img_data = str.encode(img_data.split(',')[1])
#	with open("imageToSave.jpg", "wb") as fh:
#		fh.write(base64.decodebytes(img_data))
#
#	im = Image.open('imageToSave.jpg')
#	im = im.resize((28,28))
#	im.save('imageToSave.jpg')
#
#	im = cv2.imread('imageToSave.jpg',0) 
#	#plt.imshow(im.reshape(28,28))
#	
#	threshold = 64 # to clear the noise
#	im = im.reshape(28,28) 
#	
#	for r in range(len(im)):
#		for c in range(len(im[r])):
#			im[r][c] = 0 if im[r][c] <= threshold else 1
#
#	im = im.reshape(1,28,28,1)
#	model = load_model('digit_recognizer_20200513_2330.h5')
#	res = list(model.predict_classes(im))[0]
#
#	return res
#
#form = cgi.FieldStorage()
#img_data = form.getvalue('img_data')
#print("Content-Type: text/html")    # HTML is following
#print()                             # blank line, end of headers
#print("<html><head></head><body>")
#print(predict(img_data))#this is the value to be extracted for showing on the webpage
## print(img_data)
#print("</body></html>")
#
#
#
##print(predict(img_data))
#
##"data:image/jpeg;base64,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"
















