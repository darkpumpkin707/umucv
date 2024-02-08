#!/usr/bin/env python
#https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText

import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#recorre todos los flujogramas de entrada
for _, frame in autoStream():
	H, W, _ = frame.shape
	#imagen espejo (mejor para trabajar)
	imagecv = cv.flip(frame, 1)
	#el detector necesita tener canales de color en el orden correcto
	image = cv.cvtColor(imagecv, cv.COLOR_BGR2RGB)
	results = hands.process(image)
	#cv.imshow("manos", frame)
	#cv.imshow("mirror", image)
	points = []
	#deteccion?
	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:
			#mete landmarks en un array de numpy
			for k in range(21):
				x = hand_landmarks.landmark[k].x # entre 0 y 1
				y = hand_landmarks.landmark[k].y 
				points.append([int(x*W), int(y*H)]) #para dibujan en cv
			break
		points = np.array(points) # mejor un array para poder operar matematicamente
		print(points)

		#dibujar un segmento de recta en el dedo indice
		cv.line(imagecv, points[5], points[8], color=(0,0,255), thickness = 1)
		#dibujo un circulo centrado en la palma de la mano
		center = np.mean(points[[5, 0 ,17]], axis=0)
		radio = np.linalg.norm(center-points[5])
		cv.circle(imagecv, center.astype(int), int(radio), color=(0, 255,255), thickness=3)
	cv.imshow("mirror", imagecv)
