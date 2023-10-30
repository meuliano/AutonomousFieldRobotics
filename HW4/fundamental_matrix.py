
import numpy as np
import cv2
from ransac import ransac

def compute_fundamental(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    
    return F/F[2,2]

def compute_fundamental_normalized(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))

    return F/F[2,2]


class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""
    
    def __init__(self,debug=False):
        self.debug = debug
    
    def fit(self,data):
        """ Estimate fundamental matrix using eight 
            selected correspondences. """
        
        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3,:8]
        x2 = data[3:,:8]
        
        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1,x2)
        return F
    
    def get_error(self,data,F):
        """ Compute x^T F x for all correspondences, 
            return error for each transformed point. """
        
        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        
        # Sampson distance as error measure
        Fx1 = np.dot(F,x1)
        Fx2 = np.dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = ( np.diag(np.dot(x1.T,np.dot(F,x2))) )**2 / denom 
        
        # return error per point
        return err


def F_from_ransac(x1,x2,model,maxiter=5000,match_theshold=1e-3):
    """ Robust estimation of a fundamental matrix F from point 
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).

        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    data = np.vstack((x1,x2))
    
    # compute F and return with inlier index
    F,ransac_data = ransac(data.T,model,8,maxiter,match_theshold,20,return_all=True)
    return F, ransac_data['inliers']



def getFundamentalMat(pts1,pts2):
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS,cv2.RANSAC,1.0)
    
    # only select inlier points
    pts1_new = pts1[mask.ravel()==1]
    pts2_new = pts2[mask.ravel()==1]
    return F, mask, (pts1_new, pts2_new)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2