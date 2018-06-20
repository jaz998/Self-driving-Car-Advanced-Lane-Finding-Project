############# Camera Calibration #################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from moviepy.editor import VideoFileClip

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
# def region_of_interest(img, vertices):
#     """
#     Applies an image mask.
#     """
#     # defining a blank mask to start with
#     mask = np.zeros_like(img)
#
#     # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255
#
#     # filling pixels inside the polygon defined by "vertices" with the fill color
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
#
#     # returning the image only where mask pixels are nonzero
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

#LAB Colorspace, use B channel to capture yellow line
def LABcolorspace_bChannel(img, thresh=(190,255)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    b_channel = lab[:,:,2]
    # Only normalize when there are yellows in the image
    if np.max(b_channel) > 175:
        b_channel = b_channel * (255/np.max(b_channel))
    # Apply threshold to the b_channel
    binary_output = np.zeros_like(b_channel)
    binary_output[((b_channel > thresh[0]) & (b_channel <= thresh[1]))] = 1
    return binary_output

#HLS Colorspace, use L channel to capture the white lines
def HLScolorspace_LChannel(img, thresh=(220,255)):
    # Convert the image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lChannel = hls[:,:,1]
    # Normalize
    lChannel = lChannel*(255/np.max(lChannel))
    # Applying the threshold
    binary_output = np.zeros_like(lChannel)
    binary_output[(lChannel>thresh[0]) & (lChannel<=thresh[1])] = 1
    # Return a binary image of threshold result
    return  binary_output









def pipeline(img, s_thresh=(170,255), sx_thresh=(20,100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls_l = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lab_b_channel = LABcolorspace_bChannel(img)
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
    s_binnary[(s_channel>= s_thresh[0])& (s_channel<=s_thresh[1]) | (lab_b_channel==1)] = 1

    # stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binnary))*255



    # # Defining vertices for marked area
    # imshape = img.shape
    # left_bottom = (100, imshape[0])
    # right_bottom = (imshape[1] - 20, imshape[0])
    # apex1 = (610, 410)
    # apex2 = (680, 410)
    # inner_left_bottom = (310, imshape[0])
    # inner_right_bottom = (1150, imshape[0])
    # inner_apex1 = (700, 480)
    # inner_apex2 = (650, 480)
    # vertices = np.array([[left_bottom, apex1, apex2, \
    #                       right_bottom, inner_right_bottom, \
    #                       inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
    # # Masked area
    # color_binary = region_of_interest(color_binary, vertices)
    return color_binary


road_image = cv2.imread('../test_images/test1.jpg')
#road_image = cv2.imread('C:/Users/Jason/OneDrive/Self-driving Car/Advanced Lane Finding Project/CarND-Advanced-Lane-Lines-master/test_images/straight_lines1.jpg')

road_image_size = (road_image.shape[1], road_image.shape[0])
img_size = (road_image.shape[1], road_image.shape[0])
color_binary = pipeline(road_image)
cv2.imwrite('../test_images/color_binary.png', color_binary)
print ("Image saved")
cv2.imshow('Color binary', color_binary)
cv2.waitKey()
#cv2.destroyAllWindws()

lChannel = HLScolorspace_LChannel(road_image)
cv2.imshow("L channel", lChannel)
cv2.waitKey

b_channel = LABcolorspace_bChannel(road_image)
cv2.imshow("b_channel", b_channel)
cv2.waitKey()




############ Perspective Transform ######################
# Manually find four points representing a trapezoid in color_binary while they should be two straight lines
# trapezoid points 1: 255,686; 1044,686; 831,544; 463,544;
# trapezoid points 2: 255,686; 1044,686; 682,448; 599,448;

height, width = undist.shape[:2]
print("Height:", height, " Width:", width)
offest1 = 70
# offset2 = 450
offset2 = 400

# src = np.float32([
#     (255 - offest1, 686),
#     (1044 + offest1, 686),
#     (682 + offest1, 448),
#     (599 - offest1, 448)
# ])
#
# dst = np.float32([
#     (offset2,height),
#     (width-offset2,height),
#     (width-offset2, 0),
#     (offset2,0)
# ])

# src = np.float32([
#     (255 - offest1, 686),
#     (1044 + offest1, 686),
#     (831 + offest1, 544),
#     (463 - offest1, 544)
# ])
#
# dst = np.float32([
#     (offset2,height),
#     (width-offset2,height),
#     (width-offset2, 0),
#     (offset2,0)
# ])
# offset1 = 200  # offset for dst points x value
# offset2 = 0  # offset for dst points bottom y value
# offset3 = 0  # offset for dst points top y value
# src = np.float32([[150+430,460],[1150-440,460],[1150,720],[150,720]])
# dst = np.float32([[offset1, offset3],
#                   [img_size[0] - offset1, offset3],
#                   [img_size[0] - offset1, img_size[1] - offset2],
#                   [offset1, img_size[1] - offset2]])

src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])
dst = np.float32([(450,0),
                  (width-450,0),
                  (450,height),
                  (width-450,height)])

MinV = cv2.getPerspectiveTransform(dst, src)



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
cv2.imwrite('../test_images/warped.png', warped)
cv2.imshow('warped', warped)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow('Color_binary', warped)
print("warped shape ", warped.shape)
binary_warped = warped[:,:,1]
cv2.imshow('binary_warped', binary_warped)





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
    # plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.show()

    # Measuring Curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print("Left curvature:", left_curverad, " Right Curvature:", right_curverad)



    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds


results = find_lines(binary_warped)
output_image = results[0]
left_fit = results[1]
right_fit = results[2]
ploty = results[3]
left_fitx = results[4]
right_fitx = results[5]
left_lane_inds = results[6]
right_lane_inds = results[7]

cv2.imshow('Find lane', output_image)
cv2.waitKey()


def find_lane_based_on_previous_frame (binary_warped, left_fit, right_fit, ploty = None):
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    print("left_fit[0]", left_fit[0])
    print("nonzeroy", nonzeroy)
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
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

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.show()

    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds
#
# nextFrame_results = find_lane_based_on_previous_frame(binary_warped, left_fit, right_fit)
# print("left_fit", left_fit)
# print("right_fit", right_fit)
# nextFrame_output_image = results[0]
# nextFrame_left_fit = results[1]
# nextFrame_right_fit = results[2]
# nextFrame_ploty = results[3]
# nextFrame_left_fitx = results[4]
# nextFrame_right_fitx = results[5]
# nextFrame_left_lane_inds = results[6]
# nextFrame_right_lane_inds = results[7]
# cv2.imshow("Next frame", nextFrame_output_image)
# cv2.waitKey()

############################################          Draw the lines back on the original Images ######################################

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # Was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
    def add_fit_lines(self, fit, inds):
        # Add a fit line, up to n
        if fit is not None:
            self.detected = True
            self.current_fit = fit
        else:
            self.detected = False












def draw_lines_on_original(road_image, warped, ploty, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (road_image.shape[1], road_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(road_image, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    return result




draw_on_original = draw_lines_on_original(road_image, binary_warped, ploty, left_fitx, right_fitx, MinV)
cv2.imshow("Draw on original", draw_on_original)
cv2.waitKey()

############# Process the video #####################################
left_lane = Line()
right_lane = Line()



def process_frame(frame):
    # if both left and right lanes were detected in last fame, use nextFrame function, if not, use normal processing flow
    # draw_on_original = None
    frameCount = 0
    if not left_lane.detected or not right_lane.detected:
        color_binary = pipeline(frame)
        warped = warp(color_binary, src, dst)
        binary_warped = warped[:, :, 1]
        results = find_lines(binary_warped)
        output_image = results[0]
        left_fit = results[1]
        right_fit = results[2]
        ploty = results[3]
        left_fitx = results[4]
        right_fitx = results[5]
        left_lane_inds = results[6]
        right_lane_inds = results[7]
        draw_on_original = draw_lines_on_original(frame, binary_warped, ploty, left_fitx, right_fitx, MinV)
        print("Normal process frame is called")
    else:
        color_binary = pipeline(frame)
        warped = warp(color_binary, src, dst)
        binary_warped = warped[:, :, 1]
        results = find_lane_based_on_previous_frame(binary_warped, left_lane.current_fit, right_lane.current_fit)
        output_image = results[0]
        left_fit = results[1]
        right_fit = results[2]
        ploty = results[3]
        left_fitx = results[4]
        right_fitx = results[5]
        left_lane_inds = results[6]
        right_lane_inds = results[7]
        draw_on_original = draw_lines_on_original(frame, binary_warped, ploty, left_fitx, right_fitx, MinV)
        print("Process frame that is based on the previous frame is called")
    left_lane.add_fit_lines(left_fit, left_lane_inds)
    print("left_fit ", left_fit)
    print("left_lane_inds", left_lane_inds)
    right_lane.add_fit_lines(right_fit, right_lane_inds)
    print("Frame ", frameCount)
    frameCount = frameCount + 1
    return draw_on_original


# color_binary = pipeline(frame)
    # warped = warp(color_binary, src, dst)
    # binary_warped = warped[:, :, 1]
    # results = find_lane_based_on_previous_frame(binary_warped, left_lane.current_fit, right_lane.current_fit)
    # output_image = results[0]
    # left_fit = results[1]
    # right_fit = results[2]
    # ploty = results[3]
    # left_fitx = results[4]
    # right_fitx = results[5]
    # left_lane_inds = results[6]
    # right_lane_inds = results[7]
    # draw_on_original = draw_lines_on_original(frame, binary_warped, ploty, left_fitx, right_fitx, MinV)
    # left_lane.add_fit_lines(left_fit, left_lane_inds)
    # right_lane.add_fit_lines(right_fit, right_lane_inds)






output_video = '../test_videos_output/output_video.mp4'
project_video = '../project_video.mp4'
#clip1 = VideoFileClip(project_video).subclip(0,3)
#clip1 = VideoFileClip(project_video).subclip(20, 25)
# clip1 = VideoFileClip(project_video)
# processed_clip1 = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
# processed_clip1.write_videofile(output_video, audio=False)

# result_process_frame = process_frame(road_image)
# cv2.imshow("Result processed frame", result_process_frame)
# cv2.waitKey()


































