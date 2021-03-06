import cv2
import numpy as np
import threading
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

emotions = ["negative", "neutral", "positive"]
genders = ["male", "female"]
ages = ["baby", "child", "teen", "youth", "middle", "senior"]

#load and compile model
model = load_model('../weights_h5/mobilenet_weighted_20190510-135116.h5')
opt = tf.train.AdamOptimizer()
model.compile(optimizer=opt, loss=['binary_crossentropy']*11, metrics=['accuracy'])
model.predict(np.zeros((1,224,224,3))) #to initilize model

#set to fullscren
cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Demo",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#faceDetection for openCv
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#font for text over the image
font = cv2.FONT_HERSHEY_SIMPLEX

faceSquareSize = 300
mirror = True #Mirror the video stream
cam = cv2.VideoCapture(0) #Webcam "stream"
counter = 0 #Fot prediction delay

prevFacePrediction = []
prevFacePrediction_sigmoid = []
facePredictions = []
facePredictions_color = []
lock = threading.Lock() #Lock for multi-threaded access to global variables



def getColor(value):
    red = (0,0,255)
    yellow = (0,255,255)
    green = (0,255,0)
    
    if value < 0.40:
        return red
    elif value > 0.80:
        return green
    else:
        return yellow
    

#Function for multi threaded prediction
def analyzeFace(faces_rect, img, model, Session, Graph):
    for i, (x, y, w, h) in enumerate(faces_rect):
        maxDim = max(w, h) + faceSquareSize
        xS = int((x + w/2) - maxDim/2)
        yS = int((y + h/2) - maxDim/2)
        
        position = (xS, yS)
        croppedImg = img[yS:yS+maxDim, xS:xS+maxDim] #Crop Face from image
        global facePredictions
        global facePredictions_color
        x = image.img_to_array(croppedImg)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        with Session.as_default():
            with Graph.as_default():
                preds = model.predict(x)

        lock.acquire()
        
        
        emotion = np.argmax(preds[:3])
        emotion_sigmoid = np.max(preds[:3])
        gender = np.argmax(preds[3:5])
        gender_sigmoid = np.max(preds[3:5])
        age = np.argmax(preds[5:])
        age_sigmoid = np.max(preds[5:])
        facePredictions.append([position, (emotion, gender, age)])
        facePredictions_color.append([getColor(emotion_sigmoid), getColor(gender_sigmoid), getColor(age_sigmoid)])
        lock.release()







def findClosestFace(position, facePredictions):
    minDistance = faceSquareSize*10
    minDistanceIndex = -1
    
    for i in range(0, len(facePredictions)):
        dist = np.sum(np.subtract(facePredictions[i][0], position))
        dist = abs(dist)
        if dist < minDistance:
            minDistance = dist
            minDistanceIndex = i

    return minDistanceIndex







while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)


    faces_rect = cascade.detectMultiScale(img, minNeighbors=10, minSize=(150, 150))
    maxArea = 0
    maxAreaIndex = -1
    for i, (x, y, w, h) in enumerate(faces_rect):
        area = w*h
        if maxArea < area:
            maxArea = area
            maxAreaIndex = i

    for i, (x, y, w, h) in enumerate(faces_rect):
        color = (66, 194, 244) if maxAreaIndex == i else (0, 255, 0)
        cv2.rectangle(img, (x, y - 15), (x+w, y+h+30), color, 1)
        cv2.putText(img,'Face' + str(i), (x, y - 20), font, 1, color, 2, cv2.LINE_AA)
        maxDim = max(w, h) + faceSquareSize
        xS = int((x + w/2) - maxDim/2)
        yS = int((y + h/2) - maxDim/2)
        cv2.rectangle(img, (xS, yS), (xS+maxDim, yS+maxDim), (0, 0, 255), 1)

    faces = len(faces_rect)
    if(faces > 0):
        text = ' Face Detected' if faces == 1 else ' Faces Detected'
        cv2.putText(img, str(faces) +  text,(10,30), font, 1,(0,255,0), 2, cv2.LINE_AA)

        for i, (x, y, w, h) in enumerate(faces_rect):
            maxDim = max(w, h) + faceSquareSize
            xS = int((x + w/2) - maxDim/2)
            yS = int((y + h/2) - maxDim/2)

            predictions = []
            pred_color = []
            if len(facePredictions) >= faces:
                predictions = facePredictions
                pred_color = facePredictions_color
            else:
                predictions = prevFacePrediction
                pred_color = prevFacePrediction_sigmoid

            predictIndex = findClosestFace((xS, yS), predictions)
            if predictIndex != -1:
                lock.acquire()

                em = "Emotion: " + emotions[predictions[predictIndex][1][0]]
                gen = "Gender: " + genders[predictions[predictIndex][1][1]]
                age = "Age class: " + ages[predictions[predictIndex][1][2]]
                #print("Emotion: " + emotions[predictions[predictIndex][1][0]] + " " + str(pred_color[predictIndex][0]))
                cv2.putText(img, em, (xS,yS-60), font, 0.8, pred_color[predictIndex][0], thickness=2, lineType=cv2.LINE_AA)
                
                cv2.putText(img, gen, (xS,yS-35), font, 0.8, pred_color[predictIndex][1], thickness=2, lineType=cv2.LINE_AA)
                
                cv2.putText(img, age,(xS,yS-10), font, 0.8, pred_color[predictIndex][2], thickness=2, lineType=cv2.LINE_AA)
                
                lock.release()
        
        counter += 1
        if(counter >= 30):
            counter = 0
            prevFacePrediction = facePredictions
            prevFacePrediction_sigmoid = facePredictions_color
            facePredictions = []
            facePredictions_color = []
            analyze_thread = threading.Thread(target=analyzeFace, args=[faces_rect, img, model, K.get_session(), tf.get_default_graph()])
            analyze_thread.start() #Start the prediction thread

    else:
        cv2.putText(img,'No Face Detected',(10,30), font, 1,(0,0,255), 2, cv2.LINE_AA)
        counter = 0


    cv2.imshow('Demo', img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
