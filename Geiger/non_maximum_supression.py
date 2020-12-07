# -*- coding: utf-8 -*-
"""
Reference :- 
1. https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
2. https://math.stackexchange.com/questions/87338/change-in-rotation-matrix

"""
import numpy as np
import math

def non_maximum_suppression(transformations, scores, threshR, threshT):
    """ 
    Suppress similar transformations.
    Input :
        transformations : Set of tuples, each tuple containing the rotation and the translation matrix
        scores : Scores of corresponding transformations (Defined in Global Registration step)
        threshR : Upper Threshold for theta difference between two Rotation matrices
        threshT : Upper Threshold for distance difference between two Translation matrices
        
    Output:
        list of distinct transformations.
    """
    B = []
    
    for i in range(len(transformations)):
        discard = False
        for j in range(i+1 , len(transformations)):
              
            R1 = transformations[i][0]
            R2 = transformations[j][0]        
        
            # Calculate the theta between rotation matrices of two transformation
            R12 = np.dot(R1.T , R2)
            theta = (np.trace(R12)-1)/2
            if(theta >  1):
                theta = 1
            distR = math.acos(theta)*180/math.pi
            
            T1 = transformations[i][1]
            T2 = transformations[j][1]
            
            # Calculate the distance between two translation matrices
            diff = np.subtract(T1 , T2)
            distT = np.linalg.norm(diff)
            
            similar = distR < threshR and distT < threshT
            
            # Discard the transformation if the adjacent transformation is higher in scoring
            if(similar):
                if(scores[j] > scores[i]):
                    print("DISCARDED TRANSFORMATION")
                    discard = True
                    
        if not(discard):
            B.append(transformations[i])
    return B