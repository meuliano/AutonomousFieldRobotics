import matplotlib.pyplot as plt
import cv2
import numpy as np

# HW3 Functions
def show_images(num_rows, num_columns, imgs, title=" ", isGrayscale=False):
    f, axs = plt.subplots(num_rows, num_columns)
    for i in range(num_rows):
        for j in range(num_columns):
            img_num = i*num_columns + j
            axs[i,j].imshow(imgs[img_num], cmap='gray' if isGrayscale else 'viridis')
            axs[i,j].axis('off')

    f.suptitle(title, fontsize=20)

def normalize_images(imgs):
    imgs_norm = []
    for img in imgs:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        imgs_norm.append(img_norm)
    return imgs_norm

def grayscale_images(imgs):
    imgs_gray = []
    for img in imgs:
        gr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply CLAHE to grayscale images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgs_gray.append(clahe.apply(gr))
    return imgs_gray

def find_features(sift_object, imgs_gray):
    kp = []
    des = []
    imgs_sift = []
    for img in imgs_gray:
        k, d = sift_object.detectAndCompute(img,None)
        img_sift = cv2.drawKeypoints(img, k, img, color=[255,255,0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imgs_sift.append(img_sift)
        kp.append(np.array(k))
        des.append(np.array(d))
    return kp, des, imgs_sift

def get_matches(kp1, des1, kp2, des2):
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

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    return (src_pts, dst_pts), good

def get_all_matches(imgs, print_matches=False):
    # Same Hyperparameter settings as HW3
    sift = cv2.SIFT_create(nfeatures=5000, nOctaveLayers=16, contrastThreshold=0.025, edgeThreshold=10, sigma=1.4)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    all_matches = []

    for i in range(len(imgs)-1):
        
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(imgs[i+1], cv2.COLOR_BGR2GRAY), None)

        matches = flann.knnMatch(desc_0,desc_1,k=2)

        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(desc_0, desc_1, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good.append(m)

        src_pts = np.float32([key_points_0[m.queryIdx].pt for m in good])
        dst_pts = np.float32([key_points_1[m.trainIdx].pt for m in good])
        all_matches.append((src_pts, dst_pts))

    return all_matches

    # matches = []
    # goods = []
        
    # for i in range(len(kp)-1):
    #     (pts1, pts2), good = get_matches(kp[i], des[i], kp[i+1], des[i+1])
    #     matches.append((pts1, pts2))
    #     goods.append(good)
    #     if print_matches: print("Image: ", i, ", ", i+1, " Matches: ", len(matches[i][0]))

    # return matches, goods