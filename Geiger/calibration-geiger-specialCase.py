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
            
    return planes       

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

    # cam = A
    # lidar = B

    # H = np.cov(np.dot(cam, lidar.T))

    # # find rotation
    # U, S, Vt = np.linalg.svd(H)
    # R = np.matmul(Vt.T , U.T)
    
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
    
    # print(np.linalg.det(A))
    # print(np.linalg.matrix_rank(A))
    # print(np.matmul(np.linalg.pinv(A), B))
    # print(np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T),B))
    
    translationCTL = np.linalg.lstsq(A, B, rcond=None)
    
    return translationCTL[0]

def findLidarCentres(lidar_points):
    
    lidarCentres = []
    for i in range(len(lidar_points)):
        centre = (np.sum(lidar_points[i], axis = 0)/len(lidar_points[i])).reshape((3,1))
        lidarCentres.append(centre)

    return lidarCentres

def draw_registration_result(source, target, transformation):
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def findCosineSimilarity(rotVectors,trueNormals):

    rotVectors = np.asarray(rotVectors)
    rotVectors = rotVectors.reshape((29,3))

    dists = []
    for i in range(len(rotVectors)):
        dists.append(scipy.spatial.distance.cosine(rotVectors[i], trueNormals[i]))

    return dists

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images,lidar_points = load_data()
rvecs, tvecs = get_extrinsic_matrices(images, objpoints, imgpoints, objp)

rotation_vectors = get_Rotation_Vectors(rvecs)
cameraNormals = normalInCameraFrame(rotation_vectors)
lidarNormals = normalInLidarFrame(lidar_points)

lidarNormals = np.asarray(lidarNormals)
cameraNormals = np.asarray(cameraNormals)

A=np.asarray(cameraNormals[0])
for i in range(1,len(cameraNormals)):
    A=np.hstack((A,cameraNormals[i]))

B=np.asarray(lidarNormals[0])
for i in range(1,len(lidarNormals)):
    B=np.hstack((B,lidarNormals[i]))

rotationCTL = rotateCameraToLidar(A,B)

#print(np.linalg.det(rotationCTL))
#print(np.linalg.det(np.matmul(rotationCTL.T,rotationCTL)))

cameraCentres = tvecs
lidarCentres = findLidarCentres(lidar_points)
translationCTL = translateCameraToLidar(cameraCentres, lidarCentres, lidarNormals, rotationCTL)

rotVectors = []
for cameraNormal,lidarNormal in zip(cameraNormals,lidarNormals):
    rotatedVector=np.matmul(rotationCTL,cameraNormal)
    rotVectors.append(rotatedVector)
    
cosSim = findCosineSimilarity(rotVectors,lidarNormals)
# planes = segmentation(lidar_points[0].copy(), 200, 0)