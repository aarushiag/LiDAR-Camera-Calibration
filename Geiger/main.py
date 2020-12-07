import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from icp import *
from segmentation import *
from global_registration import * 
from non_maximum_supression import *

def load_data():
    
    cwd = os.getcwd()
    path = cwd + "\\Dataset"
    files=os.listdir(path)
    
    data,images,lidar_points=[],[],[]
    image_path=path+ "\\" + files[0]
    lidar_points_path=path+ "\\" + files[1]
    images_folder=os.listdir(image_path)
    lidar_points_folder=os.listdir(lidar_points_path)
    
    for i in images_folder:
        images.append(cv2.imread(image_path + "\\" + i))
    
    for j in lidar_points_folder:
        lidar_points.append(np.load(lidar_points_path + "\\" + j))
    
    for d in data:
        images.append(d['data'][0][0][0])
        lidar_points.append(d['data'][0][0][1])
    
    return images,lidar_points   
  

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
    

def getCameraPoints(objp, rotation_vectors, tvecs, numPlanes):
    
    camera_points = []    
    for i in range(numPlanes):
        transformedPoints = []
        for j in range(len(objp)):
            point = objp[j].reshape(3,1)
            transformedPt = (np.dot(rotation_vectors[i], point) + tvecs[i]).reshape(3,)
            transformedPoints.append(transformedPt)
        camera_points.append(np.asarray(transformedPoints))
        
    return camera_points

def findCosineSimilarity(rotVectors,trueNormals):

    rotVectors = np.asarray(rotVectors)
    rotVectors = rotVectors.reshape((29,3))

    dists = []
    for i in range(len(rotVectors)):
        dists.append(scipy.spatial.distance.cosine(rotVectors[i], trueNormals[i]))

    return dists
            
# ----------------------------------- MAIN ---------------------------- #

if __name__ == '__main__':
    
    images,lidar_points = load_data()
    # The following code is commented since we assume that the data from 29 images are the planes extracted from the segmentation step
    
    # planes_lidar = segmentation(lidar_points[0].copy(), 50, 0)
    # planes_camera = segmentation(camera_points[0].copy(), 50, 0)
    

    # Calculate Camera 3D points by transforming world coordinates to camera frame of reference.
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    rvecs, tvecs = get_extrinsic_matrices(images, objpoints, imgpoints, objp)
    rotation_vectors = get_Rotation_Vectors(rvecs)
    camera_points = getCameraPoints(objp, rotation_vectors, tvecs, len(images))
    
    # Finding plane centroids
    cameraCentres = findCentres(camera_points)
    lidarCentres = findCentres(lidar_points)
    
    # Reshape array of centroids from 3D to 2D tensors
    cameraCentresShaped = np.asarray(cameraCentres)
    cameraCentresShaped = cameraCentresShaped.reshape(cameraCentresShaped.shape[0], cameraCentresShaped.shape[1])
    lidarCentresShaped = np.asarray(lidarCentres)
    lidarCentresShaped = lidarCentresShaped.reshape(lidarCentresShaped.shape[0], lidarCentresShaped.shape[1])
    
    planes_lidar = lidar_points
    planes_camera = camera_points    
    
    Z = 1
    tau = 5
    num_iterations = 100
    
    cameraNormals = {}
    lidarNormals = {}

    for i in range(len(planes_camera)):
        camNormal = findPlaneNormal(planes_camera[i])
        cameraNormals[str(planes_camera[i])] = camNormal
            
    for i in range(len(planes_lidar)):
        lidNormal = findPlaneNormal(planes_lidar[i])
        lidarNormals[str(planes_lidar[i])] = lidNormal
        
    # Calculate initial transformations
    transformations = global_registration(planes_camera, planes_lidar, cameraNormals, lidarNormals, Z, tau, num_iterations)
    
    # Refine transformations using ICP (Iterative closest Point) algorithm
    refined_transformation = []
    refined_scores = []
    
    for i in range(len(transformations)):
        rotationCTL = transformations[i][0]
        translationCTL = transformations[i][1]
        
        score = findScoreTransformation(rotationCTL, translationCTL, cameraCentres, lidarCentres)
        print("Initial :-" + score)
        
        Transformation =  np.concatenate((rotationCTL, translationCTL), axis = 1)
        B = np.asarray([0,0,0,1]).reshape(1,4)
        Transformation = np.concatenate((Transformation,B))
        Tr = icp(cameraCentresShaped, lidarCentresShaped, init_pose=Transformation, max_iterations=20, tolerance=0.001)[0]
        
        refined_rotation = Tr[0:3,0:3]
        refined_translation = Tr[0:3,3].reshape(3,1)
        
        score = findScoreTransformation(refined_rotation, refined_translation, cameraCentres, lidarCentres)
        print("Final :-" + score)
        refined_scores.append(score)
        
        refined_scores.append(score)        
        refined_transformation.append((refined_rotation, refined_translation))
     
    # Suppress the similar transformations
    refined_transformation = non_maximum_suppression(refined_transformation, refined_scores, 50, 5)

    
    # Calculate the cosine similarity between rotated vectors
    cameraNormals2 = list(cameraNormals.values())
    lidarNormals2 = list(lidarNormals.values())

    rotVectors = []
    rotationCTL = refined_transformation[2][0]
    print(np.linalg.det(rotationCTL))
    for cameraNormal,lidarNormal in zip(cameraNormals2,lidarNormals2):
        rotatedVector=np.matmul(rotationCTL,cameraNormal)
        rotVectors.append(rotatedVector)