import cv2
import numpy as np
import threading
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

#load and compile model
model = load_model('mobilenet20_20190507-165645.h5')
opt = tf.train.AdamOptimizer()
model.compile(optimizer=opt, loss=['binary_crossentropy']*11, metrics=['accuracy'])
model.predict(np.zeros((1,224,224,3))) #to initilize model

#set to fullscren
cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Demo",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#faceDetection for openCv
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#font for text over the image
font = cv2.FONT_HERSHEY_SIMPLEX

faceSquarePadding = 150
mirror = True #Mirror the video stream
cam = cv2.VideoCapture(0) #Webcam "stream"
counter = 0 #Fot prediction delay

emotion = 0
gender = 0
age = 0
lock = threading.Lock() #Lock for multi-threaded access to global variables

#Function for multi threaded prediction
def analyzeFace(face, model, Session, Graph, position):
    global emotion, gender, age, predictionData
    x = image.img_to_array(face)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    with Session.as_default():
        with Graph.as_default():
            preds = model.predict(x)

    lock.acquire()
    emotion = str(np.argmax(preds[:3]))
    gender = str(np.argmax(preds[3:5]))
    age = str(np.argmax(preds[5:]))
    lock.release()
    print("Emotion: " + str(np.argmax(preds[:3])))
    print("Gender: " + str(np.argmax(preds[3:5])))
    print("Age class: " + str(np.argmax(preds[5:])))


while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)


    faces_rect = cascade.detectMultiScale(img, minNeighbors=10, minSize=(150, 150))[:1]
    for i, (x, y, w, h) in enumerate(faces_rect):
        cv2.rectangle(img, (x, y - 15), (x+w, y+h+30), (0, 255, 0), 1)
        cv2.putText(img,'Face' + str(i), (x, y - 20), font, 1,(0,255,0), 2, cv2.LINE_AA)
        maxDim = max(w, h) + faceSquarePadding
        xS = int((x + w/2) - maxDim/2)
        yS = int((y + h/2) - maxDim/2)
        cv2.rectangle(img, (xS, yS), (xS+maxDim, yS+maxDim), (0, 0, 255), 1)

    faces = len(faces_rect)
    if(faces > 0):
        text = ' Face Detected' if faces == 1 else ' Faces Detected'
        cv2.putText(img, str(faces) +  text,(10,30), font, 1,(0,255,0), 2, cv2.LINE_AA)

        lock.acquire()
        cv2.putText(img, "Emotion: " + str(emotion),(xS,yS-60), font, 0.8,(0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img, "Gender: " + str(gender),(xS,yS-35), font, 0.8,(0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img, "Age class: " + str(age),(xS,yS-10), font, 0.8,(0,0,255), 1, cv2.LINE_AA)
        lock.release()
        
        counter += 1
        if(counter >= 30):
            counter = 0
            print("Time to predict")
            predictionData = []
            for i, (x, y, w, h) in enumerate(faces_rect):
                maxDim = max(w, h) + faceSquarePadding
                xS = int((x + w/2) - maxDim/2)
                yS = int((y + h/2) - maxDim/2)
                
                croppedImg = img[yS:yS+maxDim, xS:xS+maxDim] #Crop Face from image
                analyze_thread = threading.Thread(target=analyzeFace, args=[croppedImg, model, K.get_session(), tf.get_default_graph(), (xS, yS)])
                analyze_thread.start() #Start the prediction thread

    else:
        cv2.putText(img,'No Face Detected',(10,30), font, 1,(0,0,255), 2, cv2.LINE_AA)
        counter = 0


    cv2.imshow('Demo', img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
