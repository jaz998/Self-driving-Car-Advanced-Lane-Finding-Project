############# Camera Calibration #################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# prepare object points
nx = 9 #number of inside corners in x
ny = 6 #number of inside corners in y

# prepare object points
objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in read world space
imgpoints = [] # 2d points in image plane.

# Read a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')
print(images)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    print(ret)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()

test_image = cv2.imread('../camera_cal/calibration2.jpg')
test_image_size = (test_image.shape[1], test_image.shape[0])

# Do image calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, test_image_size, None, None)

undist = cv2.undistort(test_image, mtx, dist, None, mtx)
cv2.imwrite('../camera_cal/test_undist2.jpg', undist)

# Save the image calibration results for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('../camera_cal/disk_pickle2.p', 'wb'))

################## Color/gradient threshold ###################
def pipeline(img, s_thresh=(170,255), sx_thresh=(20,100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # take the derivative in x
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel>= sx_thresh[0])&(scaled_sobel<= sx_thresh[1])] = 1

    # Threshold color channel
    s_binnary = np.zeros_like(s_channel)
    s_binnary[(s_channel>= s_thresh[0])& (s_channel<=s_thresh[1])] = 1
    # stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binnary))*255
    return color_binary


road_image = cv2.imread('../test_images/straight_lines1.jpg')
road_image_size = (road_image.shape[1], road_image.shape[0])
color_binary = pipeline(road_image)
cv2.imwrite('../test_images/color_binary.png', color_binary)
print ("Image saved")
#cv2.imshow('Color binary', color_binary)
#cv2.waitKey()
#cv2.destroyAllWindws()


############ Perspective Transform ######################
# Manually find four points representing a trapezoid in color_binary while they should be two straight lines
# trapezoid points 1: 255,686; 1044,686; 831,544; 463,544;
# trapezoid points 2: 255,686; 1044,686; 682,448; 599,448;

height, width = undist.shape[:2]
print("Height:", height, " Width:", width)
offest1 = 0
# offset2 = 450
offset2 = 400

src = np.float32([
	(255 - offest1, 686),
	(1044 + offest1, 686),
	(682 + offest1, 448),
	(599 - offest1, 448)
])

dst = np.float32([
	(offset2,height),
	(width-offset2,height),
	(width-offset2, 0),
	(offset2,0)
])

def warp(img, src, dst):
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
	return warped


#src = np.float32([(255,686), (1044,686), (831, 544), (463,544)])
#dst = np.float32([(255,686), (1044,686), (1044,544), (255,544)])

#src = np.float32([(255,686), (1044,686), (682, 448), (599,448)])
#dst = np.float32([(255,686), (1044,686), (1044,448), (255,448)])

# src points in the order of top left, top right, bottom right, bottom left
# src = np.float32([(599,448), (682, 448), (1044,686), (255,686)])

# M = cv2.getPerspectiveTransform(src, dst)
# warped = cv2.warpPerspective(color_binary, M, (color_binary.shape[1], color_binary.shape[0]))
warped = warp(color_binary, src, dst)
# cv2.imwrite('../test_images/warped.png', warped)
# cv2.imshow('warped', warped)
# cv2.waitKey()
#cv2.destroyAllWindows()
# cv2.imshow('Color_binary', warped)



################# Detect Lane Line #############################
# histogram = np.sum(warped[warped.shape[0]//2:,:], axis = 0)
# plt.plot(histogram)
# plt.show()


















