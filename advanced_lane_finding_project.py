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


road_image = cv2.imread('../test_images/test2.jpg')
#road_image = cv2.imread('C:/Users/Jason/OneDrive/Self-driving Car/Advanced Lane Finding Project/CarND-Advanced-Lane-Lines-master/test_images/straight_lines1.jpg')

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
# cv2.destroyAllWindows()
# cv2.imshow('Color_binary', warped)
print("warped shape ", warped.shape)
binary_warped = warped[:,:,1]






################# Detect Lane Line #############################
# histogram = np.sum(warped[warped.shape[0]//2:,:], axis = 0)
# plt.plot(histogram)
# plt.show()



############################    Finding the Lines ################

def find_lines(binary_warped):
	out_img = []

	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
	print('out_img ', out_img)
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0] // 2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0] // nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
					  (0, 255, 0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
					  (0, 255, 0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
						  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
						   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	return out_img


result = find_lines(binary_warped)
cv2.imshow('result', result)
cv2.waitKey()


















