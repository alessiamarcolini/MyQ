import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

model = load_model('mobilenet20_20190507-165645.h5')
opt = tf.train.AdamOptimizer()
model.compile(optimizer=opt, loss=['binary_crossentropy']*11, metrics=['accuracy'])

cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Demo",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

mirror = True
cam = cv2.VideoCapture(0)
counter = 0

emotion = 0
gender = 0
age = 0

while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)


    faces_rect = cascade.detectMultiScale(img, minNeighbors=3, minSize=(150, 150))
    for i, (x, y, w, h) in enumerate(faces_rect):
        cv2.rectangle(img, (x, y - 15), (x+w, y+h+30), (0, 255, 0), 1)
        cv2.putText(img,'Face' + str(i), (x, y - 20), font, 1,(0,255,0), 2, cv2.LINE_AA)
        maxDim = max(w, h) + 100
        xS = int((x + w/2) - maxDim/2)
        yS = int((y + h/2) - maxDim/2)
        cv2.rectangle(img, (xS, yS), (xS+maxDim, yS+maxDim), (0, 0, 255), 1)


    faces = len(faces_rect)
    if(faces > 0):
        text = ' Face Detected' if faces == 1 else ' Faces Detected'
        cv2.putText(img, str(faces) +  text,(10,30), font, 1,(0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, "Emotion: " + str(emotion),(xS,yS-50), font, 0.8,(0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img, "Gender: " + str(gender),(xS,yS-30), font, 0.8,(0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img, "Age class: " + str(age),(xS,yS-10), font, 0.8,(0,0,255), 1, cv2.LINE_AA)
        
        counter += 1
        if(counter >= 30):
            print("Time to predict")
            for i, (x, y, w, h) in enumerate(faces_rect):
                maxDim = max(w, h) + 100
                xS = int((x + w/2) - maxDim/2)
                yS = int((y + h/2) - maxDim/2)
                
                croppedImg = img[yS:yS+maxDim, xS:xS+maxDim] #Crop Face from image
                x = image.img_to_array(croppedImg)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)
                
                emotion = str(np.argmax(preds[:3]))
                gender = str(np.argmax(preds[3:5]))
                age = str(np.argmax(preds[5:]))
                print("Emotion: " + str(np.argmax(preds[:3])))
                print("Gender: " + str(np.argmax(preds[3:5])))
                print("Age class: " + str(np.argmax(preds[5:])))
                
                counter = 0
    else:
        cv2.putText(img,'No Face Detected',(10,30), font, 1,(0,0,255), 2, cv2.LINE_AA)
        counter = 0


    cv2.imshow('Demo', img)
    key = cv2.waitKey(1)
    if key == 27:
        break



cv2.destroyAllWindows()
