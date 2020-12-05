# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 02:25:07 2020

@author: dell
"""

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
from segmentation import *

# ----------------------------------- SECTION B ---------------------------- #

def global_registration(planes_camera, planes_lidar, normal_cam, normal_lidar, Z, tau, num_iterations):
    
    cam_triples = getTriples(planes_camera)
    lidar_triples = getTriples(planes_lidar)
    
    cam_weights = calc_weights(cam_triples, Z, normal_cam)
    lidar_weights = calc_weights(lidar_triples, Z, normal_lidar)
    
    isSelected = {}
    
    transformations = []
    scores = []
    
    counter = 0
    while(counter < num_iterations):
                      
        cam_tpl = random.choices(cam_triples, weights = cam_weights, k = 1)[0]
        lidar_tpl = random.choices(lidar_triples, weights = lidar_weights, k = 1)[0]
                
        while(str((lidar_tpl,cam_tpl)) in isSelected.keys()):
            cam_tpl = random.choices(cam_triples, weights = cam_weights, k = 1)[0]
            lidar_tpl = random.choices(lidar_triples, weights = lidar_weights, k = 1)[0]
            
        isSelected[str((lidar_tpl,cam_tpl))] = 1
        
        cameraNormals = []
        lidarNormals = []

        for i in cam_tpl:
            camNormal = normal_cam[str(i)]
            cameraNormals.append(camNormal)
            
        for i in lidar_tpl:
            lidNormal = normal_lidar[str(i)]
            lidarNormals.append(lidNormal)
        
        lidarNormals = np.asarray(lidarNormals)
        cameraNormals = np.asarray(cameraNormals)
        
        A=np.asarray(cameraNormals[0])
        for i in range(1,len(cameraNormals)):
            A=np.hstack((A,cameraNormals[i]))
        
        B=np.asarray(lidarNormals[0])
        for i in range(1,len(lidarNormals)):
            B=np.hstack((B,lidarNormals[i]))
                    
        cameraPoints = [cam_tpl[0], cam_tpl[1], cam_tpl[2]]
        lidar_points = [lidar_tpl[0], lidar_tpl[1], lidar_tpl[2]]
        
        cameraCentres = findCentres(cameraPoints)
        lidarCentres = findCentres(lidar_points)
        
        rotationCTL = rotateCameraToLidar(A , B)
        translationCTL = translateCameraToLidar(cameraCentres, lidarCentres, lidarNormals, rotationCTL)
        
        transformations.append((rotationCTL, translationCTL))
        
        score = findScoreTransformation(rotationCTL, translationCTL, cameraCentres, lidarCentres)
        scores.append(score)
        
        counter += 1
        
    final_transformation = []
    max_score = max(scores)
    lower_threshold = tau * max_score
    
    for i in range(len(scores)):
        if(scores[i] > lower_threshold):
            final_transformation.append(transformations[i])
            
    return final_transformation

def get_extrinsic_matrices(images, objpoints, imgpoints, objp):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for img in images:
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
 
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners = np.reshape(corners, (48,2))
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return rvecs, tvecs 

def get_Rotation_Vectors(rvecs):

    rvecs = np.asarray(rvecs)
    rotation_vectors = []

    for i in range(len(rvecs)):
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvecs[i], rotation_matrix)
        rotation_vectors.append(rotation_matrix)

    return rotation_vectors

def normalInCameraFrame(rotation):

    normalInWorld = [0,0,1]
    normalInWorld = np.reshape(normalInWorld, (3,1))
    cameraNormals = []
    
    for i in range(len(rotation)):
        
        vector = np.matmul(rotation[i],normalInWorld)
        cameraNormals.append(vector)

    return cameraNormals

def normalInLidarFrame(lidar_points):

    normals = []

    for i in range(len(lidar_points)):
        
        point = lidar_points[i]
        point = np.subtract(point , np.mean(point, axis = 0))
        avg = np.linalg.norm(point[0])
        for j in range(1, point.shape[0]):
            avg += np.linalg.norm(point[j])  
        avg = avg/point.shape[0]
        point = point / avg  
        
        variables = np.concatenate((point[:,0:2] , np.ones((np.shape(point)[0],1),dtype=int)), axis = 1)
        rhs = point[:,2]
        rhs = rhs.reshape(np.shape(rhs)[0],1)
        
        x = np.linalg.lstsq(variables, rhs, rcond=None)
        normal = np.concatenate((x[0][0:2], -1 * np.ones((1,1))))

        modVal = math.sqrt(math.pow(normal[0],2) + math.pow(normal[1],2) + math.pow(normal[2],2))
        normal = normal/modVal
        
        # If n.p0 where p0 is any point lying on the plane < 0 , this means n is in the direction of the plane, otherwise in the opposite direction.
        # We need this step since by fixing c = -1, we are fixing the direction of c. Hence we must check the direction of normal with respect to the plane.

        sign = np.matmul(normal.T,lidar_points[i][0].reshape((3,1)))
        if(sign<0):
            normal = 0-normal
            
        normals.append(normal)
        
    return normals

def rotateCameraToLidar(A,B):
    
    num_rows, num_cols = A.shape;

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    centroid_A = centroid_A.reshape((3,1))
    centroid_B = centroid_B.reshape((3,1))

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = np.matmul(Am ,Bm.T)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T , U.T)
        
    return R

def translateCameraToLidar(cameraCentres, lidarCentres, lidarNormals, rotationCTL):

    N0 = np.dot(lidarNormals[0], lidarNormals[0].T)
    cam = np.dot(N0, np.dot(rotationCTL, cameraCentres[0]))
    lidar = np.dot(N0, lidarCentres[0])
    
    A = N0
    
    for i in range(1, len(lidarCentres)):
        N = np.dot(lidarNormals[i], lidarNormals[i].T)
        cam += np.dot(N, np.dot(rotationCTL, cameraCentres[i]))
        lidar += np.dot(N, lidarCentres[i])
        A+= N        
        
    B = lidar - cam
   
    translationCTL = np.linalg.lstsq(A, B, rcond=None)
    
    return translationCTL[0]

def findCentres(points):
    centres = []
    for i in range(len(points)):
        centre = (np.sum(points[i], axis = 0)/len(points[i])).reshape((3,1))
        centres.append(centre)
    return centres

def getTriples(planes):
    n = len(planes)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((planes[i], planes[j], planes[k]))
    return triples

def calc_weights(triples, Z, normals):
    weights = []
    for i in range(len(triples)):
        tpl = triples[i]
        na = normals[str(tpl[0])]
        nb = normals[str(tpl[1])]
        nc = normals[str(tpl[2])]
        w = math.exp(- np.dot(na.T, nb)[0][0] - np.dot(na.T, nc)[0][0] - np.dot(nb.T, nc)[0][0])
        w = w/Z
        weights.append(w)
    return weights

def findNearest(transformedPoint, points):
    
    diff = np.subtract(transformedPoint, points[0])
    min_dist = np.linalg.norm(diff)
    closest = points[0]
    
    for i in range(1, len(points)):
        diff = np.subtract(transformedPoint, points[0])
        dist = np.linalg.norm(diff)
        
        if(dist < min_dist):
            dist = min_dist
            closest = points[i]
            
    return closest

def findScoreTransformation(rotationCTL, translationCTL, cameraCentres, lidarCentres):
    
    score = 0
    for i in range(len(cameraCentres)):
        transformedCentre = np.add(np.matmul(rotationCTL, cameraCentres[i]) , translationCTL)
        lidarClosest = findNearest(transformedCentre, lidarCentres)
        diff = np.subtract(transformedCentre, lidarClosest)
        distance =  np.linalg.norm(diff)
        score -= distance
    
    return score  