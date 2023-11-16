
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

def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_P_from_essential(E):
    """ Computes the second camera matrix (assuming P1 = [I 0])
    from an essential matrix. Output is a list of four
    possible camera matrices. See section 9.6.2 from H&Z book (p.258)."""
    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # return all four solutions
    P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
    np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
    np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
    np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]
    return P2

def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from
    least squares solution. """
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def triangulate(x1,x2,P1,P2):
    """ Two-view triangulation of points in
    x1,x2 (3*n homog. coordinates). """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points dont match.")
    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return np.array(X).T

class Camera(object):
    def __init__(self,P):
        self.P = P
        
    def project(self,X):
        x = np.dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]
        return x

def kpsObjToArray(kp):
    kps = np.array([p.pt for p in kp])
    kps_rad = np.array([p.size / 2 for p in kp]) # rad==scale
    kps = np.hstack([kps,kps_rad[:,np.newaxis]])
    return kps

def findNeighbours(kp1, kp2, des1, des2, numNeighbors=1):
    # FLANN parameters https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    flann_match = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    good = []

    # ratio test as per Lowe's paper
    for m,n in flann_match:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.array([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])
    return src_pts, dst_pts

def plotMatches(ax, im1, im2, kps1, kps2, pairs):
    ax.imshow(np.hstack((im1, im2)), cmap="gray")
    t = np.arange(0, 2*np.pi, 0.1)

    # Display matches
    colors = ['C{}'.format(i) for i in range(10)]

    for k in range(pairs.shape[0]):
        ic = k % 10
        
        pid1 = pairs[k, 0]
        pid2 = pairs[k, 1]
    
        loc1 = kps1[int(pid1),0:2]
        r1 = kps1[int(pid1), 2] * 3 # Treble the radius for seeing the keypoints better
        loc2 = kps2[int(pid2),0:2]
        r2 = kps2[int(pid2), 2] * 3

        ax.plot(loc1[0]+r1*np.cos(t), loc1[1]+r1*np.sin(t), 'c-', linewidth=1)
        ax.plot(loc2[0]+r2*np.cos(t)+im1.shape[1], loc2[1]+r2*np.sin(t), 'c-', linewidth=1)
        ax.plot([loc1[0], loc2[0]+im1.shape[1]], [loc1[1], loc2[1]], color='{}'.format(colors[ic]), linestyle='-')

def get_fundamental_matrix(pts1,pts2):
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS,cv2.RANSAC,1.0)
    
    # only select inlier points
    pts1_new = pts1[mask.ravel()==1]
    pts2_new = pts2[mask.ravel()==1]
    return F, mask, (pts1_new, pts2_new)

def draw_lines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1.astype(int)),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2.astype(int)),5,color,-1)
    return img1,img2