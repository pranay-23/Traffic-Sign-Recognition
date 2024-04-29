from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2
from flask import Flask, render_template, Response, request
import datetime, time
import os, sys
from threading import Thread


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
global capture,switch 
capture=0
grey=0
neg=0
face=0
switch=2
rec=0
threshold = 0.90     # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_PATH ='model.h5'

model = load_model(MODEL_PATH)

try:
    os.mkdir('shots')
except OSError as error:
    pass

net = cv2.dnn.readNetFromCaffe('saved_model/deploy.prototxt.txt', 'saved_model/res10_300x300_ssd_iter_140000.caffemodel')


camera = cv2.VideoCapture(1)


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        img = np.asarray(frame)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        # cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(frame, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        # PREDICT IMAGE
        predictions = model.predict(img)
        # classIndex = model.predict_classes(img)
        classIndex=np.argmax(predictions)
        probabilityValue = np.amax(predictions)
        if probabilityValue > threshold:
        #print(getCalssName(classIndex))
            if classIndex!=15 and classIndex!=13:
                cv2.putText(frame,str(getClassName2(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow("Result", frame)
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p,frame)
                
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


# @app.route('/')
# def index():
#     return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
           
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1 or switch==2):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(1)
                switch=1                   
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h - Specifies a maximum speed of 20 kilometers per hour.'
    elif classNo == 1: return 'Speed Limit 30 km/h - Specifies a maximum speed of 30 kilometers per hour.'
    elif classNo == 2: return 'Speed Limit 50 km/h - Specifies a maximum speed of 50 kilometers per hour.'
    elif classNo == 3: return 'Speed Limit 60 km/h - Specifies a maximum speed of 60 kilometers per hour.'
    elif classNo == 4: return 'Speed Limit 70 km/h - Specifies a maximum speed of 70 kilometers per hour.'
    elif classNo == 5: return 'Speed Limit 80 km/h - Specifies a maximum speed of 80 kilometers per hour.'
    elif classNo == 6: return 'End of Speed Limit 80 km/h - Indicates the end of the speed limit of 80 kilometers per hour.'
    elif classNo == 7: return 'Speed Limit 100 km/h - Specifies a maximum speed of 100 kilometers per hour.'
    elif classNo == 8: return 'Speed Limit 120 km/h - Specifies a maximum speed of 120 kilometers per hour.'
    elif classNo == 9: return 'No passing - Prohibits overtaking other vehicles.'
    elif classNo == 10: return 'No passing for vehicles over 3.5 metric tons - Prohibits overtaking for heavy vehicles.'
    elif classNo == 11: return 'Right-of-way at the next intersection - Indicates priority at the next intersection.'
    elif classNo == 12: return 'Priority road - Indicates priority for vehicles on this road.'
    elif classNo == 13: return 'Yield - Requires vehicles to give the right-of-way to traffic on the intersecting road.'
    elif classNo == 14: return 'Stop - Requires vehicles to come to a complete stop.'
    elif classNo == 15: return 'No vehicles - Prohibits entry of all vehicles.'
    elif classNo == 16: return 'Vehicles over 3.5 metric tons prohibited - Restricts heavy vehicles.'
    elif classNo == 17: return 'No entry - Prohibits entry of all vehicles.'
    elif classNo == 18: return 'General caution - Indicates a general caution for drivers.'
    elif classNo == 19: return 'Dangerous curve to the left - Warns of a dangerous curve to the left.'
    elif classNo == 20: return 'Dangerous curve to the right - Warns of a dangerous curve to the right.'
    elif classNo == 21: return 'Double curve - Warns of a double curve ahead.'
    elif classNo == 22: return 'Bump Ahead - Indicates a bumpy road surface.'
    elif classNo == 23: return 'Slippery road - Warns of slippery road conditions.'
    elif classNo == 24: return 'Road narrows on the right - Indicates a narrowing road ahead.'
    elif classNo == 25: return 'Road work - Warns of road construction or maintenance.'
    elif classNo == 26: return 'Traffic signals - Indicates the presence of traffic signals ahead.'
    elif classNo == 27: return 'Pedestrians - Warns of pedestrians crossing the road.'
    elif classNo == 28: return 'Children crossing - Warns of children crossing the road.'
    elif classNo == 29: return 'Bicycles crossing - Warns of bicycles crossing the road.'
    elif classNo == 30: return 'Beware of ice/snow - Warns of icy or snowy road conditions.'
    elif classNo == 31: return 'Wild animals crossing - Warns of wild animals crossing the road.'
    elif classNo == 32: return 'End of all speed and passing limits - Marks the end of speed and passing restrictions.'
    elif classNo == 33: return 'Turn right ahead - Indicates a right turn ahead.'
    elif classNo == 34: return 'Turn left ahead - Indicates a left turn ahead.'
    elif classNo == 35: return 'Ahead only - Vehicles must proceed straight ahead.'
    elif classNo == 36: return 'Go straight or right - Vehicles may proceed straight or turn right.'
    elif classNo == 37: return 'Go straight or left - Vehicles may proceed straight or turn left.'
    elif classNo == 38: return 'Keep right - Vehicles must keep to the right of the road.'
    elif classNo == 39: return 'Keep left - Vehicles must keep to the left of the road.'
    elif classNo == 40: return 'Roundabout mandatory - Indicates mandatory movement through a roundabout.'
    elif classNo == 41: return 'End of no passing - Marks the end of a no-passing zone.'
    elif classNo == 42: return 'End of no passing by vehicles over 3.5 metric tons - Marks the end of a no-passing zone for heavy vehicles.'

def getClassName2(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bump Ahead'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # PREDICT IMAGE
    # predictions = model.predict(img)
    # classIndex = model.predict_classes(img)
    # probabilityValue =np.amax(predictions)
    predicted_probabilities = model.predict(img)  # Get predicted probabilities for each class
    classIndex = np.argmax(predicted_probabilities)  # Get the index of the class with highest probability
    probabilityValue =np.amax(predicted_probabilities)
    if probabilityValue > 0.80:
        preds = getClassName(classIndex)
        return preds
    else:
        return "Cannot Detect the Sign"

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)


camera.release()
cv2.destroyAllWindows()     
