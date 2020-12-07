# This code is originally written by Aarushi Agarwal (2016216)

import numpy as np
import cv2
import glob
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.spatial.distance
import random

def get_extrinsic_matrices(images, objpoints, imgpoints):

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

def normalAndDistancesInCameraFrame(rotation,translation):

    normalInWorld = [0,0,1]
    normalInWorld = np.reshape(normalInWorld, (3,1))
    cameraNormals = []
    distances = []
    
    for i in range(len(rotation)):
        
        vector = np.matmul(rotation[i],normalInWorld)
        cameraNormals.append(vector)

        #dist = np.matmul(vector.T,translation[i])[0][0]
        dist = math.sqrt(tvecs[i][0]**2 + tvecs[i][1]**2 + tvecs[i][2]**2)
        distances.append(dist)

    return cameraNormals,distances

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

def normalAndDistancesInLidarFrame(lidar_points):

    normals = []
    distances = []

    for i in range(len(lidar_points)):
        point = lidar_points[i]
    
        variables = np.concatenate((point[:,0:2] , np.ones((np.shape(point)[0],1),dtype=int)), axis = 1)
        rhs = point[:,2]
        rhs = rhs.reshape(np.shape(rhs)[0],1)

        x = np.linalg.lstsq(variables, rhs)
        #x = np.matmul(np.linalg.inv(np.matmul(variables.T,variables)),np.matmul(variables.T,rhs))
        normal = np.concatenate((x[0][0:2], -1 * np.ones((1,1))))

        modVal = math.sqrt(math.pow(normal[0],2) + math.pow(normal[1],2) + math.pow(normal[2],2))
        normal = normal/modVal
        
        # If n.p0 where p0 is any point lying on the plane < 0 , this means n is in the direction of the plane, otherwise in the opposite direction.
        # We need this step since by fixing c = -1, we are fixing the direction of c. Hence we must check the direction of normal with respect to the plane.

        sign = np.matmul(normal.T,point[0].reshape((3,1)))
        if(sign<0):
            normal = 0-normal
            
        normals.append(normal)

        d = (x[0][2]/modVal)[0]
        if(d<0):
            d = -d
        distances.append(d)
        
    return normals,distances

def rotateLidarToCamera(A,B):

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

    """# special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = np.matmul(Vt.T , U.T)"""
    
   # transformation = np.matmul(np.matmul(B,A.T),np.linalg.inv(np.matmul(A,A.T)))
    
    return R

def translateLidarToCamera(cameraDist, lidarDist, lidarNormals, rotationLTC):

    A = []
    b = []
    
    for i in range(len(cameraDist)):
        A.append(lidarNormals[i].T)
        b.append(lidarDist[i] - cameraDist[i])

    A = np.asarray(A)
    b = np.asarray(b)

    A = A.reshape((A.shape[0],3))

    x = np.linalg.lstsq(A, b)
    x = -x[0]
    x = np.matmul(rotationLTC, x)
    
    return x

def findNewVectors(rotationLTC, lidarNormals):
    
    lidarNormals = np.asarray(lidarNormals)
    lidarNormals = lidarNormals.reshape(np.shape(lidarNormals)[0], 3)
    return np.matmul(rotationLTC,lidarNormals)

def findCosineSimilarity(rotVectors,cameraNormals):

    rotVectors = np.asarray(rotVectors)
    rotVectors = rotVectors.reshape((29,3))
    meanRotatedVector = np.mean(rotVectors , axis = 0)

    dists = []
    for i in range(len(rotVectors)):
        dists.append(scipy.spatial.distance.cosine(rotVectors[i], cameraNormals[i]))

    return dists

def randomSampling(k ,cameraDist, lidarDist, lidarNormals):
    population = np.arange(0, 29, 1).tolist()
    randomPop = random.sample(population,k)

    sampledCameraDist = []
    sampledLidarDist = []
    sampledLidarNormals = []

    for i in range(len(randomPop)):
        sampledCameraDist.append(cameraDist[randomPop[i]])
        sampledLidarDist.append(lidarDist[randomPop[i]])
        sampledLidarNormals.append(lidarNormals[randomPop[i]])

    return sampledCameraDist, sampledLidarDist, sampledLidarNormals

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images,lidar_points = load_data()
rvecs, tvecs = get_extrinsic_matrices(images, objpoints, imgpoints)

rotation_vectors = get_Rotation_Vectors(rvecs)
cameraNormals, cameraDist= normalAndDistancesInCameraFrame(rotation_vectors,tvecs)
lidarNormals,lidarDist = normalAndDistancesInLidarFrame(lidar_points)

lidarNormals = np.asarray(lidarNormals)
cameraNormals = np.asarray(cameraNormals)

A=np.asarray(cameraNormals[0])
for i in range(1,len(cameraNormals)):
    A=np.hstack((A,cameraNormals[i]))

print(A)

B=np.asarray(lidarNormals[0])
for i in range(1,len(lidarNormals)):
    B=np.hstack((B,lidarNormals[i]))

rotationLTC = rotateLidarToCamera(B,A)
translationLTC = translateLidarToCamera(cameraDist, lidarDist, lidarNormals, rotationLTC)

print(np.linalg.det(rotationLTC))
print(np.linalg.det(np.matmul(rotationLTC.T,rotationLTC)))

rotVectors = []
for cameraNormal,lidarNormal in zip(cameraNormals,lidarNormals):
    rotatedVector=np.dot(rotationLTC,lidarNormal)
    rotVectors.append(rotatedVector)

cosSim = findCosineSimilarity(rotVectors,cameraNormals)

translationLTCSampled = []
for i in range(0,1000):
    sampledCameraDist, sampledLidarDist, sampledLidarNormals = randomSampling(15, cameraDist, lidarDist, lidarNormals)
    sampledTranslation = translateLidarToCamera(sampledCameraDist, sampledLidarDist, sampledLidarNormals, rotationLTC)
    translationLTCSampled.append(sampledTranslation)
    
mean = np.mean(translationLTCSampled, axis = 0)
std = np.std(translationLTCSampled,axis = 0)
upperBound = mean + std
lowerBound = mean - std

# Uncomment the following code piece to have a look at the visualizations

"""index=0
for cameraNormal,lidarNormal in zip(cameraNormals,lidarNormals):
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    rotatedVector=np.matmul(rotationLTC,lidarNormal)
    #x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),np.arange(-0.8, 1, 0.2),np.arange(-0.8, 1, 0.8))
    ax.quiver(0,0,0,cameraNormal[0],cameraNormal[1],cameraNormal[2],length=0.1,normalize=True,color="blue")
    ax.quiver(0,0,0,lidarNormal[0],lidarNormal[1],lidarNormal[2],length=0.1,normalize=True,color="red")
    #ax.quiver(pts[0][0],pts[0][1],pts[0][2],normal_camera_approx[0],normal_camera_approx[1],normal_camera_approx[2],length=0.1,normalize=True,color="green")
    ax.quiver(0,0,0,rotatedVector[0],rotatedVector[1],rotatedVector[2],length=0.1,normalize=True,color="green")
    plt.show()
    if(index == 5):
        break
    index+=1"""
