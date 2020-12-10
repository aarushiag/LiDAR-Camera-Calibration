import numpy as np
import math
from planefitting import*


# ----------------------------------- SECTION A ---------------------------- #

def segmentation(points, k, threshold):
    
    """ Calculates the set of points which represents a single plane in the point cloud data.
    Input :
        points: A (n x 3) ndarray of total 3D points captured in a scene
        k: Amount of neighbours required to calculate the normal of a point
        threshold: Lower threshold of allowed similarity between normals of two points in a single plane
        
    Output :
        list of planes, each plain is a list of 3D points """
        
    normals = {}
    neighbours = {}
    
    #Calculate normal for each point by considering a plane of K nearest points
    for i in range(len(points)):
        N = findNearestK(points, i, k)
        n = findPlaneRansac(N)
        normals[str(points[i])] = n   
        neighbours[str(points[i])] = N
        
    planes = []    
    planes.append([points[0]])
    points = np.delete(points, 0, axis = 0)
    
    #Club all the points into different segments (Planes)
    while(len(points) > 0):
        
        plane = planes[len(planes)-1]
        nrj = normals[str(plane[0])]  #Seed's normal
        indexes = [] 
         
        #Iterate over all the points to include those which have similar normal to the seed's normal
        for i in range(len(points)):
            neighbour = neighbours[str(points[i])]
            nri  = normals[str(points[i])]
            counter = 0
            for j in range(len(neighbour)):
                if(isPresent(neighbour[j] , plane) and (abs(np.dot(nri.T, nrj)[0][0]) > threshold)):
                    counter = 1
                    break
                
            if(counter == 1):
                plane.append(points[i])
                indexes.append(i)
            
        #Remove points which are included in the segment
        newlist = [v for i, v in enumerate(points) if i not in indexes]
        points = np.asarray(newlist)
        
        #Add new plane with a new seed
        if(len(points) > 0):
            planes.append([points[0]])
            points = np.delete(points, 0, axis = 0)
            
    for i in range(len(planes)):
        planes[i] = np.asarray(planes[i])
            
    return planes

def findNearestK(points, index, k):
    
    """ Finds the k nearest point to a point at the given index in the point cloud.
    Input :
        point : Point cloud
        index : Index of the point whose neighbours are required
        k : Number of neighbours
        
    Output :
        A (k x 3) ndarray denoting the k neighbour coordinates in 3d space """
    
    neighbour = []
    point1 = points[index]
    
    for i in range(len(points)):
        if (i != index):
            point2 = points[i]
            diff = np.subtract(point1, point2)
            dist = np.linalg.norm(diff)
            neighbour.append((point2, dist))
            
    # Sort neighbours on the basis of distance from the point
    neighbour.sort(key = lambda x: x[1]) 
    neighbour = neighbour[0:k+1]
    
    N = []
    for i in range(k):
        N.append(neighbour[i][0])
    N.append(point1)
    N = np.asarray(N)
    return N

def findPlaneRansac(pts):
    
    n = pts.shape[0]
    max_iterations = 100
    goal_inliers = n * 0.3

    # test data
    xyzs = pts

    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    
    normal = np.asarray([a,b,c]).reshape((3,1))
    modVal = math.sqrt(math.pow(normal[0],2) + math.pow(normal[1],2) + math.pow(normal[2],2))
    normal = normal/modVal
    
    sign = np.matmul(normal.T, pts[0].reshape((3,1)))
    if(sign<0):
        normal = 0-normal
        
    return normal
    

def findPlaneNormal(pts):
    
    """ Finds the normal of a plane with the given 3D points.
    Input : A (n x 3) ndarray with n points
    Output : A (3 x 1) vector denoting the normal of the plane """
    
    # Normalize the point cloud
    point = np.subtract(pts , np.mean(pts, axis = 0))
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

    sign = np.matmul(normal.T, pts[0].reshape((3,1)))
    if(sign<0):
        normal = 0-normal
        
    return normal

def isPresent(elem, lst):
    
    """ Checks whether an element (of any type like a vector) is present in the list or not """
    
    for i in range(len(lst)):
        if (str(elem) == str(lst[i])):
            return True
    return False