

import pygame
import imutils
import time
import numpy as np
import os
import time
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector" , "model.caffemodel"])

detectFace = cv2.dnn.readNet(prototxtPath, weightsPath)


maskModel = load_model("training_model.h5")



alarm = False

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def detect_and_find_roi(frame, detectFace, maskModel):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	detectFace.setInput(blob)
	detections = detectFace.forward()


	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		prob = detections[0, 0, i, 2]


		if prob > 0.5:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:

		preds = maskModel.predict(faces)


	return (locs, preds)




print("video stream is starting")
vs = VideoStream(src=0).start()
time.sleep(3.0)


while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	(locs, preds) = detect_and_find_roi(frame, detectFace, maskModel)


	for (box, pred) in zip(locs, preds):

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred


		if mask >withoutMask:
			label = "Mask"
			alarm = False
		else:
			label = "no Mask"
			alarm=True

		if label =="Mask":
			color = (255, 255,255)
		else:
			color = (0, 0, 255)


		label = "{}:".format(label)



		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		if(alarm == True):
			cv2.putText(frame,"Put on your Mask !!",(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.60 , (0,255,255),3 )
		else:
			cv2.putText(frame,"Nice Job",(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.60 , (255,255,255),3 )

	cv2.imshow("Frame", frame)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
