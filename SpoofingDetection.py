# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:55:28 2019

@author: AsadAbbas
"""
from sklearn.externals import joblib
import numpy as np
import cv2

class SpoofDetection:
    def __init__(self,model_path):
        try:
            self.clf = joblib.load(model_path)
        except IOError as e:
            print ("Error loading model <"+model_path+">: {0}".format(e.strerror))
            exit(0)
            
    def __calc_hist(self,img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)
    
    def spoof_check(self,img):
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

        ycrcb_hist = self.__calc_hist(img_ycrcb)
        luv_hist = self.__calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = self.clf.predict_proba(feature_vector)
        prob = prediction[0][1]
        print("Prob",prob)
        if 0 != prob:
            status = False
            if np.mean(prob) >= 0.1:
                status = True
                
        return status