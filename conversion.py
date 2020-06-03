# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:38:36 2019

@author: AsadAbbas
"""
from tensorflow import lite
import pickle
import hickle as hkl
from sklearn.externals import joblib
class ConvertToTfLite:
    def __init__(self):
        pass
    
    def from_pkl(pkl_path,tf_lite_path=None):
        pkl_mdl=joblib.load(pkl_path)
        hkl.dump(pkl_mdl,pkl_path.replace(pkl_path.split('.')[-1],'h5'), "w")
        tfconverter=lite.TFLiteConverter.from_keras_model_file(pkl_path.replace(pkl_path.split('.')[-1],'h5'))
#        tfconverter=lite.TFLiteConverter.from_saved_model(pkl_path)
        tflite_model=tfconverter.convert()
        if tf_lite_path:
            open(tf_lite_path,'wb').write(tflite_model)
        else:
            open(pkl_path.replace(pkl_path.split('.')[-1],'tflite'),'wb').write(tflite_model)
    def from_pbgraph(graph_path,tf_lite_path=None):
        tfconverter=lite.TFLiteConverter.from_frozen_graph(graph_path,
                                                           input_arrays=["Preprocessor/sub"],
                                                           output_arrays=["concat","concat_1"],
                                                           input_shapes={"Preprocessor/sub":[1,300,300,3]}
                                                           )
        tflite_model=tfconverter.convert()
        if tf_lite_path:
            open(tf_lite_path,'wb').write(tflite_model)
        else:
            open(graph_path.replace(graph_path.split('.')[-1],'tflite'),'wb').write(tflite_model)
if __name__=='__main__':
    converter=ConvertToTfLite
#    converter.from_pkl('D:/Asad/DeepLearning/FaceObject/spoofdetection/spoofing_detection/trained_models/replay_attack_trained_models/replay-attack_ycrcb_luv_extraTreesClassifier.pkl')
    converter.from_pbgraph('D:/Asad/DeepLearning/FaceObject/tensorflowmodels/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')
    