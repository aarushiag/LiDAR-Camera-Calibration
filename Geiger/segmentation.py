import numpy as np
import cv2
import glob
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.spatial.distance
import random
import open3d as o3d
import copy
import scipy 

# ----------------------------------- SECTION A ---------------------------- #

def segmentation(points, k, threshold):
    normals = {}
    neighbours = {}
    
    for i in range(len(points)):
        N = findNearestK(points, i, k)
        n = findPlaneNormal(N)
        normals[str(points[i])] = n
        neighbours[str(points[i])] = N
        
    planes = []    
    planes.append([points[0]])
    points = np.delete(points, 0, axis = 0)
    
    while(len(points) > 0):
        
        plane = planes[len(planes)-1]
        nrj = normals[str(plane[0])]
        indexes = []
         
        for i in range(len(points)):
            neighbour = neighbours[str(points[i])]
            nri  = normals[str(points[i])]
            counter = 0
            for j in range(len(neighbour)): 
                if(isPresent(neighbour[j] , plane) and (np.dot(nri.T, nrj)[0][0] > threshold)):
                    counter = 1
                    break
                
            if(counter == 1):
                plane.append(points[i])
                indexes.append(i)
            
        newlist = [v for i, v in enumerate(points) if i not in indexes]
        points = np.asarray(newlist)
        
        if(len(points) > 0):
            planes.append([points[0]])
            points = np.delete(points, 0, axis = 0)
            
    for i in range(len(planes)):
        planes[i] = np.asarray(planes[i])
            
    return planes

def findNearestK(points, index, k):
    
    neighbour = []
    point1 = points[index]
    
    for i in range(len(points)):
        if (i != index):
            point2 = points[i]
            diff = np.subtract(point1, point2)
            dist = np.linalg.norm(diff)
            neighbour.append((point2, dist))
            
    neighbour.sort(key = lambda x: x[1]) 
    neighbour = neighbour[0:k+1]
    
    N = []
    for i in range(k):
        N.append(neighbour[i][0])
    N = np.asarray(N)
    return N

def findPlaneNormal(pts):
    
    variables = np.concatenate((pts[:,0:2] , np.ones((np.shape(pts)[0],1),dtype=int)), axis = 1)
    rhs = pts[:,2]
    rhs = rhs.reshape(np.shape(rhs)[0],1)
        
    x = np.linalg.lstsq(variables, rhs, rcond=None)
    normal = np.concatenate((x[0][0:2], -1 * np.ones((1,1))))

    modVal = math.sqrt(math.pow(normal[0],2) + math.pow(normal[1],2) + math.pow(normal[2],2))
    normal = normal/modVal
        
    # If n.p0 where p0 is any point lying on the plane < 0 , this means n is in the direction of the plane, otherwise in the opposite direction.
    # We need this step since by fixing c = -1, we are fixing the direction of c. Hence we must check the direction of normal with respect to the plane.

    sign = np.matmul(normal.T, pts[0].reshape((3,1)))
    if(sign<0):
        normal = 0-normal
        
    return normal

def isPresent(elem, lst):
    for i in range(len(lst)):
        if (str(elem) == str(lst[i])):
            return True
    return False