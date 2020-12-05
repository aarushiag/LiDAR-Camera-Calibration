import numpy as np
import cv2
import glob
import os
import math
import matplotlib.pyplot as plt
import scipy.spatial.distance
import random
import operator as op
from functools import reduce
from sklearn.neighbors import NearestNeighbors
from icp import *
from segmentation import *
from global_registration import * 

def load_data():
    
    cwd = os.getcwd()
    path = cwd + "\\Dataset"
    files=os.listdir(path)
    
    data,images,lidar_points=[],[],[]
    image_path=path+ "\\" + files[0]
    lidar_points_path=path+ "\\" + files[1]
    images_folder=os.listdir(image_path)
    lidar_points_folder=os.listdir(lidar_points_path)
    
    os.chdir(image_path)
    for i in images_folder:
        images.append(cv2.imread(i))
    
    os.chdir(lidar_points_path)
    
    for j in lidar_points_folder:
        lidar_points.append(np.load(j))
    
    for d in data:
        images.append(d['data'][0][0][0])
        lidar_points.append(d['data'][0][0][1])
    
    return images,lidar_points      

# ----------------------------------- MAIN ---------------------------- #

if __name__ == '__main__':
    
    images,lidar_points = load_data()
    planes_lidar = segmentation(lidar_points[0].copy(), 50, 0)
    
    camera_points = lidar_points
    planes_camera = segmentation(camera_points[0].copy(), 50, 0)
    
    Z = 1
    tau = 5
    num_iterations = 15
    
    cameraNormals = {}
    lidarNormals = {}

    for i in range(len(planes_camera)):
        camNormal = findPlaneNormal(planes_camera[i])
        cameraNormals[str(planes_camera[i])] = camNormal
            
    for i in range(len(planes_lidar)):
        lidNormal = findPlaneNormal(planes_lidar[i])
        lidarNormals[str(planes_lidar[i])] = lidNormal
        
    transformations = global_registration(planes_camera, planes_lidar, cameraNormals, lidarNormals, Z, tau, num_iterations)
    refined_transformation = []
    
    for i in range(len(transformations)):
        rotationCTL = transformations[i][0]
        translationCTL = transformations[i][1]
        Transformation =  np.concatenate((rotationCTL, translationCTL), axis = 1)
        B = np.asarray([0,0,0,1]).reshape(1,4)
        Transformation = np.concatenate((Transformation,B))
        Tr = icp(camera_points[0], lidar_points[0], init_pose=Transformation, max_iterations=20, tolerance=0.001)[0]
        
        refined_rotation = Tr[0:3,0:3]
        refined_translation = Tr[0:3,3].reshape(3,1)
        refined_transformation.append(((refined_rotation, refined_translation)))
        
