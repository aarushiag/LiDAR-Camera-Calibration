# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import random
from functools import reduce
from sklearn.neighbors import NearestNeighbors
from segmentation import *

# ----------------------------------- SECTION B ---------------------------- #

def global_registration(planes_camera, planes_lidar, normal_cam, normal_lidar, Z, tau, num_iterations):
    
    """ Calculates the transformation between camera planes and lidar planes extracted from the segmentation stage.
    Input :
        planes_camera : List of segments in Camera point cloud.
        planes_lidar : List of segments in Lidar point cloud.
        normal_cam : Dictionary of normals mapped to the corresponding planes in camera frame.
        normal_lidar : Dictionary of normals mapped to the corresponding planes in lidar frame.
        Z : Normalizing constant for selecting the random plane triple.
        tau : fraction of maximum score that can be allowed for a transformation to be included in final set.
        num_iterations : Number of times the Algorithm 1 shall be implemented (= Number of transformations generated)
        
    Output :
        List of refined transformations """
    
    # Form all the possible triples from the planes
    cam_triples = getTriples(planes_camera)
    lidar_triples = getTriples(planes_lidar)
    
    # Calculate the probability weighst for each triple
    cam_weights = calc_weights(cam_triples, Z, normal_cam)
    lidar_weights = calc_weights(lidar_triples, Z, normal_lidar)
    
    # Dictionary storing the triples which are already selected while calculating transformation
    isSelected = {}
    
    transformations = []
    scores = []
    
    counter = 0
    while(counter < num_iterations):
               
        print("Iteration :- " + str(counter))
        
        cam_tpl = random.choices(cam_triples, weights = cam_weights, k = 1)[0]
        
        #Choose the corresponding lidar planes for the chosen planes in camera triple 
        #(This is only used when we have corresponding lidar data, 
        #if not then uncomment the commented and choose the lidar triple randomly)
        
        lidar_tpl = (planes_lidar[deep_index(planes_camera , cam_tpl[0])], 
                    planes_lidar[deep_index(planes_camera , cam_tpl[1])],
                    planes_lidar[deep_index(planes_camera , cam_tpl[2])])
        
        #lidar_tpl = random.choices(lidar_triples, weights = lidar_weights, k = 1)[0]
                
        while(str((lidar_tpl,cam_tpl)) in isSelected.keys()):
            cam_tpl = random.choices(cam_triples, weights = cam_weights, k = 1)[0]
            # lidar_tpl = random.choices(lidar_triples, weights = lidar_weights, k = 1)[0]
            lidar_tpl = (planes_lidar[deep_index(planes_camera , cam_tpl[0])], 
                    planes_lidar[deep_index(planes_camera , cam_tpl[1])],
                    planes_lidar[deep_index(planes_camera , cam_tpl[2])])
            
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
        
        # Change normals from a 3D vector to 2D vector
        A=np.asarray(cameraNormals[0])
        for i in range(1,len(cameraNormals)):
            A=np.hstack((A,cameraNormals[i]))
        
        B=np.asarray(lidarNormals[0])
        for i in range(1,len(lidarNormals)):
            B=np.hstack((B,lidarNormals[i]))
                    
        cameraPoints = [cam_tpl[0], cam_tpl[1], cam_tpl[2]]
        lidar_points = [lidar_tpl[0], lidar_tpl[1], lidar_tpl[2]]
        
        # Find the centroids of the planes to calculate the transformation
        cameraCentres = findCentres(cameraPoints)
        lidarCentres = findCentres(lidar_points)
        
        rotationCTL = rotateCameraToLidar(A , B)
        translationCTL = translateCameraToLidar(cameraCentres, lidarCentres, lidarNormals, rotationCTL)
        transformations.append((rotationCTL, translationCTL))
        
        # Calculate the score of the transformation as defined in the paper
        score = findScoreTransformation(rotationCTL, translationCTL, cameraCentres, lidarCentres)
        scores.append(score)
        
        counter += 1
        
    final_transformation = []
    max_score = max(scores)
    lower_threshold = tau * max_score
    
    # Refine the transformation by choosing the higher scoring transformations
    for i in range(len(scores)):
        if(scores[i] > lower_threshold):
            print(scores[i])
            final_transformation.append(transformations[i])
            
    return final_transformation

def deep_index(lst, sub_list):
    """ Find the index of a 2D vector in a list of 2D vectors"""
    for i in range(len(lst)):
        if((lst[i] == sub_list).all()):
            return i 

def rotateCameraToLidar(A,B):
    
    """ Calculate the rotation matrix from A vectors to B vectors
    Input :
        A : (n x 3) ndarray of normals in a particular frame with n images.
        B : (n x 3) ndarray of normals in a another frame with n images.
        
    Output :
        R : (3 x 3) Rotation matrix such that AR = B"""
        
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
        
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[num_cols-1,:] *= -1
       R = np.dot(Vt.T, U.T)
       
    return R

def translateCameraToLidar(cameraCentres, lidarCentres, lidarNormals, rotationCTL):
    
    """ Calculate the translation matrix from camera frame to lidar frame of reference
    Input :
        cameraCentres : List of (3 x 1) ndarray denoting the centroid of the camera planes.
        lidarCentres : List of (3 x 1) ndarray denoting the centroid of the lidar planes.
        lidarNormals : (n x 3 x 1) ndarray of normals in a another frame with n images.
        rotationCTL : (3 x 3) rotation matrix.
        
    Output :
        T : (3 x 1) Translation matrix such that P1*R + T = P2"""

    # Solved by partial differentiation of the equation given in the paper (for minimizing)
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
    
    """ Find centroid of a plane """
    
    centres = []
    for i in range(len(points)):
        centre = (np.sum(points[i], axis = 0)/len(points[i])).reshape((3,1))
        centres.append(centre)
    return centres

def getTriples(planes):
    
    """ Generate all the possible triplets from the planes. """
    
    n = len(planes)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((planes[i], planes[j], planes[k]))
    return triples

def calc_weights(triples, Z, normals):
    
    """ Calculate the probability weights of each triple
    Input :
        triples : List of possible triples
        Z : Normalizing constant for probability function
        normals : Disctionary of planes mapped to their normals
    Output :
        List of weights for the corresponding triples"""
        
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
    
    """ Find the nearest point to a transformed point in the transformed point cloud
    Input :
        transformedPoint : Point from a frame (camera/lidar) transformed into another frame (lidar / camera)
    Output :
        Coordinate of the closest point in the destination frame
        """        
    diff = np.subtract(transformedPoint, points[0])
    min_dist = np.linalg.norm(diff)
    closest = points[0]
    
    for i in range(1, len(points)):
        diff = np.subtract(transformedPoint, points[i])
        dist = np.linalg.norm(diff)
        
        if(dist < min_dist):
            min_dist = dist
            closest = points[i]
            
    return closest

def findScoreTransformation(rotationCTL, translationCTL, cameraCentres, lidarCentres):
    
    """ Find the score of a transformation as mentioned in the paper
    Input :
        rotationCTL :  (3 x 3) Rotation matrix
        translationCTL : (3 x 1) translation matrix
        cameraCentres : Source points (Centroid of camera plane)
        lidarCentres : Destination points (Centroid of lidar plane)"""
    score = 0
    for i in range(len(cameraCentres)):
        transformedCentre = np.add(np.matmul(rotationCTL, cameraCentres[i]) , translationCTL)
        lidarClosest = findNearest(transformedCentre, lidarCentres)
        diff = np.subtract(transformedCentre, lidarClosest)
        distance =  np.linalg.norm(diff)
        score -= distance
    
    return score  