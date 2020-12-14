import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy
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

def randomSampling(total_pop, k ,cameraCentres, lidarCentres, lidarNormals):
    population = np.arange(0, total_pop, 1).tolist()
    randomPop = random.sample(population,k)

    sampledCameraCentres = []
    sampledLidarCentres = []
    sampledLidarNormals = []

    for i in range(len(randomPop)):
        sampledCameraCentres.append(cameraCentres[randomPop[i]])
        sampledLidarCentres.append(lidarCentres[randomPop[i]])
        sampledLidarNormals.append(lidarNormals[randomPop[i]])

    validation_CameraCentres = []
    validation_LidarCentres = []
    validation_LidarNormals = []
    for i in range(total_pop):
        if(i not in randomPop):
            validation_CameraCentres.append(cameraCentres[i])
            validation_LidarCentres.append(lidarCentres[i])
            validation_LidarNormals.append(lidarNormals[i])
            
    return sampledCameraCentres, sampledLidarCentres, sampledLidarNormals, validation_CameraCentres, validation_LidarCentres, validation_LidarNormals

def newScore(rotationCTL, translationCTL, lidarNormals, lidarCentres, cameraCentres):
    
    lidarNor = list(lidarNormals.values())
    
    transformed = []
    for i in range(len(cameraCentres)):
        transformedCentre = np.add(np.matmul(rotationCTL, cameraCentres[i]) , translationCTL)
        transformed.append(transformedCentre)
    
    lidar = np.dot(lidarNor[0].T , lidarCentres[0])[0]
    for i in range(1, len(lidarCentres)):
        lidar += np.dot(lidarNor[i].T , lidarCentres[i])[0]
    lidar = lidar / len(lidarCentres)
    
    camera = np.dot(lidarNor[0].T , transformed[0])[0]
    for i in range(1, len(cameraCentres)):
        # lidarClosest = findNearest(transformed[i], lidarCentres)
        index = i
        # for j in range(len(lidarCentres)):
        #     if(lidarCentres[j] == lidarClosest).all():
        #        index = j 
        camera += np.dot(lidarNor[index].T , transformed[i])[0]
    camera = camera / len(cameraCentres)
    
    return(abs(lidar - camera))
            
# ----------------------------------- MAIN ---------------------------- #

if __name__ == '__main__':
    
    images,lidar_points = load_data()
    # The following code is commented since we assume that the data from 29 images are the planes extracted from the segmentation step
    
    # xyzs = np.concatenate((lidar_points[0],lidar_points[22]), axis = 0)
    # planes_lidar = segmentation(xyzs.copy(), 200, 0.5)
    # planes_camera = segmentation(camera_points[0].copy(), 50, 0)
    

    # Calculate Camera 3D points by transforming world coordinates to camera frame of reference.
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    objp = objp*0.108 #Scaling factor
    
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
    num_iterations = 500
    
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
    refined_scores_normalized = []
    
    for i in range(len(transformations)):
        rotationCTL = transformations[i][0]
        translationCTL = transformations[i][1]
        
        score = findScoreTransformation(rotationCTL, translationCTL, cameraCentres, lidarCentres)
        score = score/len(cameraCentres)
        print("Initial Normalized Score : " + str(score))
               
        Transformation =  np.concatenate((rotationCTL, translationCTL), axis = 1)
        B = np.asarray([0,0,0,1]).reshape(1,4)
        Transformation = np.concatenate((Transformation,B))
        Tr = icp(cameraCentresShaped, lidarCentresShaped, init_pose=Transformation, max_iterations=100, tolerance=0.001)[0]
        
        refined_rotation = Tr[0:3,0:3]
        refined_translation = Tr[0:3,3].reshape(3,1)
        
        score = findScoreTransformation(refined_rotation, refined_translation, cameraCentres, lidarCentres)
        score = score/len(cameraCentres)
        print("Final Normalized Score after ICP : " + str(score))
        
        refined_scores_normalized.append(score.round(3))
        refined_transformation.append((refined_rotation, refined_translation))
     
    # Suppress the similar transformations
    refined, discarded = non_maximum_suppression(refined_transformation, refined_scores_normalized, 50, 5)
    print(newScore(refined[0][0], refined[0][1], lidarNormals, lidarCentres, cameraCentres))
    
    
    #Code for validating Rotation Vector [Calculate the cosine similarity between rotated vectors]
    """cameraNormals2 = list(cameraNormals.values())
    lidarNormals2 = list(lidarNormals.values())

    rotVectors = []
    rotationCTL = refined_transformation[2][0]
    print(np.linalg.det(rotationCTL))
    for cameraNormal,lidarNormal in zip(cameraNormals2,lidarNormals2):
        rotatedVector=np.matmul(rotationCTL,cameraNormal)
        rotVectors.append(rotatedVector)
        
    cosSim = findCosineSimilarity(rotVectors,lidarNormals2)
    print(cosSim)"""
    
    # Code for validating translation over testing Set
    """testing_scores = []
    translationCTLSampled = []
    for i in range(100):
        sampledCameraCentres, sampledLidarCentres, sampledLidarNormals, validation_CameraCentres, validation_LidarCentres, validation_LidarNormals = randomSampling(len(images), 15 ,cameraCentres, lidarCentres, lidarNormals2)
        sampledTranslationCTL = translateCameraToLidar(sampledCameraCentres, sampledLidarCentres, sampledLidarNormals, rotationCTL)
        score = findScoreTransformation(rotationCTL, sampledTranslationCTL, validation_CameraCentres, validation_LidarCentres)
        score = score / len(validation_LidarCentres)
        testing_scores.append(score)
        translationCTLSampled.append(sampledTranslationCTL)    
    print(testing_scores)"""
    
    #Code for visualizing Rotation vector
    """index=0
    for cameraNormal,lidarNormal in zip(cameraNormals,lidarNormals):
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        rotatedVector=np.matmul(rotationCTL,cameraNormal)
        #x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),np.arange(-0.8, 1, 0.2),np.arange(-0.8, 1, 0.8))
        ax.quiver(0,0,0,cameraNormal[0],cameraNormal[1],cameraNormal[2],length=0.1,normalize=True,color="blue")
        ax.quiver(0,0,0,lidarNormal[0],lidarNormal[1],lidarNormal[2],length=0.1,normalize=True,color="red")
        #ax.quiver(pts[0][0],pts[0][1],pts[0][2],normal_camera_approx[0],normal_camera_approx[1],normal_camera_approx[2],length=0.1,normalize=True,color="green")
        ax.quiver(0,0,0,rotatedVector[0],rotatedVector[1],rotatedVector[2],length=0.1,normalize=True,color="green")
        plt.show()
        if(index == 5):
            break
        index+=1"""
        
      
    # Code for validating translation matrix
    """ import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    
    rotationCTL = refined_transformation[2][0]
    translationCTL = refined_transformation[2][1]
    lidarNormals = list(lidarNormals.values())
    
    transformed = []
    for i in range(len(cameraCentres)):
        transformedCentre = np.add(np.matmul(rotationCTL, cameraCentres[i]) , translationCTL)
        transformed.append(transformedCentre)
    
    transformed = np.asarray(transformed)
    transformed = transformed.reshape((transformed.shape[0],3))
    lidarCentres = np.asarray(lidarCentres)
    lidarCentres = lidarCentres.reshape((lidarCentres.shape[0],3))
    
    ax.scatter3D(lidarCentres.T[0], lidarCentres.T[1], lidarCentres.T[2], marker = 'o')
    ax.scatter3D(transformed.T[0], transformed.T[1], transformed.T[2], marker = '^')"""