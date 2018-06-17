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


# The below order_points function and four_point_transform functions are inspired by Adrian Rosebrock article 4 Point OpenCV getPerspective Transform Example (link: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, src):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = src
	print("Order points are ", rect)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	print("MaxWeidth:", maxWidth, " MaxHeight:", maxHeight)

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


# Manually find four points representing a trapezoid in color_binary while they should be two straight lines
# 255,686; 1044,686; 831,544; 463,544;
# 255,686; 1044,686; 682,448; 599,448;


#src = np.float32([(255,686), (1044,686), (831, 544), (463,544)])
#dst = np.float32([(255,686), (1044,686), (1044,544), (255,544)])

#src = np.float32([(255,686), (1044,686), (682, 448), (599,448)])
#dst = np.float32([(255,686), (1044,686), (1044,448), (255,448)])

# src points in the order of top left, top right, bottom right, bottom left
src = np.float32([(599,448), (682, 448), (1044,686), (255,686)])

# M = cv2.getPerspectiveTransform(src, dst)
# warped = cv2.warpPerspective(color_binary, M, (color_binary.shape[1], color_binary.shape[0]))
warped = four_point_transform(road_image	, src)
cv2.imwrite('../test_images/warped.png', warped)
cv2.imshow('warped', warped)
cv2.waitKey()
cv2.destroyAllWindows()




################# Detect Lane Line #############################













