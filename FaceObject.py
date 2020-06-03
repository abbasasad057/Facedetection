# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:22:57 2019

@author: AsadAbbas
"""
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import time as tt
from time import time
import face_recognition
from SpoofingDetection import SpoofDetection
#import argparse
import pickle
import hickle as hkl
import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
import requests
#%%
class FaceObject(SpoofDetection):
    def __init__(self,objectDetectionAlgo,encoding_path):
        super().__init__('./replay-attack_ycrcb_luv_extraTreesClassifier.pkl')
        print("Loading Object Detection Model")        
        self.net = model_zoo.get_model(objectDetectionAlgo, pretrained=True)
#        print(self.net)
        print("Loading Face Recognition Model")
#        self.data = pickle.loads(open(encoding_path, "rb").read())
        self.data= hkl.load('encodings.h5')
    
    def recognize_faces(self,face_image,detection_method):
        boxes = face_recognition.face_locations(face_image,model=detection_method)
        encodings = face_recognition.face_encodings(face_image, boxes)
        names = []
        for encoding in encodings:
            matches,face_scores = face_recognition.compare_faces(self.data["encodings"],encoding)
#            face_scores=np.array(1-face_scores)
#            print("matches",matches)
            print("scores",max([1-fs for fs in face_scores]))
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in set(matchedIdxs):
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)  
        return boxes,[face_id for name in names for face_id,face_name in enumerate(set(self.data["names"]+["Unknown"])) if name==face_name]#,face_scores
    
    def transform_face_bboxes(self,xmin,ymin,face_bbox):
            return xmin+face_bbox[0][3],ymin+face_bbox[0][0],face_bbox[0][1]+xmin,face_bbox[0][2]+ymin
    
    def detect_objects(self,im_fname):           
        x, img = data.transforms.presets.ssd.load_test(im_fname, short=355)
#        print(img.shape)
#        print(self.spoof_check(img))
        if self.spoof_check(img):
            thresh=0.5
            print("[INFO] Detecting Objects...")
            class_IDs, scores, bounding_boxs = self.net(x)
            classes=self.net.classes
            bboxes=bounding_boxs[0].asnumpy()
            face_bboxes=[]
            face_ids=[]
            print("[INFO] Recognizing Faces...")
            object_bboxes=[]
            person_scores=[]
            for bbox_indx,bbox in enumerate(bboxes):
                cls_id = int(class_IDs[0].asnumpy().flat[bbox_indx]) if class_IDs[0].asnumpy() is not None else -1
                if classes[cls_id]=='person'and scores[0].asnumpy().flat[bbox_indx] > thresh:
                    object_bboxes.append(bbox)
                    person_scores.append(scores[0].asnumpy().flat[bbox_indx])
#                    print('Score',scores[0].asnumpy().flat[bbox_indx])
                    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
                    crp_img=img[ymin:ymax,xmin:xmax]
#                    plt.imshow(crp_img)
                    face_bbox,face_id=self.recognize_faces(crp_img,"cnn")
                    if len(face_bbox)>0:
                        face_bboxes.append([self.transform_face_bboxes(xmin,ymin,face_bbox)])
                        face_ids.append(face_id)
                    else:
                        continue
#            print("Persons=",len(object_bboxes))
#            print("PersonsNp=",len(np.asarray(object_bboxes)))
            return img,np.asarray(object_bboxes),np.asarray(person_scores),14*np.ones(len(object_bboxes)),face_bboxes,face_ids
        else:
#            print("Not running")
            return img,None,None,None,None,None
    
    def detect_object_video(self,im_frame):
        x, img = data.transforms.presets.ssd.load_test_video(frame=im_frame, short=355)
#        print(img.shape)
#        print(self.spoof_check(img))
        if self.spoof_check(img):
            thresh=0.5
            print("[INFO] Detecting Objects...")
            class_IDs, scores, bounding_boxs = self.net(x)
            classes=self.net.classes
            bboxes=bounding_boxs[0].asnumpy()
            face_bboxes=[]
            face_ids=[]
            print("[INFO] Recognizing Faces...")
            object_bboxes=[]
            person_scores=[]
            for bbox_indx,bbox in enumerate(bboxes):
                cls_id = int(class_IDs[0].asnumpy().flat[bbox_indx]) if class_IDs[0].asnumpy() is not None else -1
                if classes[cls_id]=='person'and scores[0].asnumpy().flat[bbox_indx] > thresh:
                    object_bboxes.append(bbox)
                    person_scores.append(scores[0].asnumpy().flat[bbox_indx])
#                    print('Score',scores[0].asnumpy().flat[bbox_indx])
                    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
                    crp_img=img[ymin:ymax,xmin:xmax]
#                    plt.imshow(crp_img)
                    face_bbox,face_id=self.recognize_faces(crp_img,"cnn")
                    if len(face_bbox)>0:
                        face_bboxes.append([self.transform_face_bboxes(xmin,ymin,face_bbox)])
                        face_ids.append(face_id)
                    else:
                        continue
#            print("Persons=",len(object_bboxes))
#            print("PersonsNp=",len(np.asarray(object_bboxes)))
            return img,np.asarray(object_bboxes),np.asarray(person_scores),14*np.ones(len(object_bboxes)),face_bboxes,face_ids
        else:
#            print("Not running")
            return img,None,None,None,None,None
        
    def plot_bboxes(self,img,object_bboxes,object_scores,object_class_ids,face_bboxes,face_ids):
        ax = utils.viz.plot_bbox(img, object_bboxes, object_scores,object_class_ids, class_names=self.net.classes)
        ax = utils.viz.plot_bbox(img=img,bboxes=np.asarray([f[0] for f in face_bboxes]),labels=np.asarray([f_id[0] for f_id in face_ids]),class_names=list(set(self.data["names"]+["Unknown"])),ax=ax)
        plt.show()

if __name__=='__main__':
    t1=time()
    detect=FaceObject("ssd_512_resnet50_v1_voc","encodings_new.h5")
    t3=time()
    print("Time to load models",(t3-t1)/60)
    url='http://192.168.43.1:8080/shot.jpg'
#    cap = cv2.VideoCapture(0)
#    vs = VideoStream(src=0).start()
    writer = None
    tt.sleep(2.0)
    prevTime = 0
    print('Starting video stream')
    while True:
            imgResp=requests.get(url,stream=True)
#            frame = vs.read()
#            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#            rgb = imutils.resize(frame, width=750)
            imgNp=np.array(bytearray(imgResp.raw.read()),dtype=np.uint8)
#            imgNp=np.array(bytearray(frame,dtype=np.uint8)
            frame=cv2.imdecode(imgNp,-1)
            frame = cv2.resize(frame, (800,600), fx=10, fy=10)
#            frame = cv2.resize(imgResp, (800,600), fx=10, fy=10)
#            cv2.imwrite('example.jpg',frame)
            img,object_bboxes,object_scores,object_class_ids,face_bboxes,face_ids=detect.detect_object_video(im_frame=frame) #detect.detect_objects("examples/IMAG0316.jpg")
            try:
                detect.plot_bboxes(img,object_bboxes,object_scores,object_class_ids,face_bboxes,face_ids)
            except TypeError:
                continue
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#        print("Type Error")
#        plt.imshow(img)
#    t2=time()
#    print("Time for recognizing objects and faces",(t2-t3)/60)
#    print("Total time:",(t2-t1)/60)
#