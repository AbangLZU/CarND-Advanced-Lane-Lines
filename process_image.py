from __future__ import print_function

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline


# image distortion correction and
# returns the undistorted image
def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def subplot(rows, cols, imgs):
    nums = rows * cols
    fig = plt.figure(1, figsize=(16, 9))
    for i in range(1, nums+1):
        plt.subplot(rows, cols, i)
        plt.imshow(imgs[i-1])


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, channel='S', thresh=(90, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'S':
        X = hls[:, :, 2]
    elif channel == 'H':
        X = hls[:, :, 0]
    elif channel == 'L':
        X = hls[:, :, 1]
    else:
        print('illegal channel !!!')
        return
    binary_output = np.zeros_like(X)
    binary_output[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary_output


def r_select(img, thresh=(200, 255)):
    R = img[:,:,0]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary

def combine_filters(img):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 255))
    l_binary = hls_select(img, channel='L', thresh=(40, 255))
    s_binary = hls_select(img, channel='S', thresh=(120, 255))
    combined_lsx = np.zeros_like(gradx)
    combined_lsx[((l_binary == 1) & (s_binary == 1) |(gradx == 1))] = 1
    return combined_lsx

def perspective_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def find_line_fit(wrap_img, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(wrap_img[wrap_img.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((wrap_img, wrap_img, wrap_img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(wrap_img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = wrap_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = wrap_img.shape[0] - (window + 1) * window_height
        win_y_high = wrap_img.shape[0] - window * window_height
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

    # to plot
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, out_img

# Generate x and y values for plotting
def get_fit_xy(wrap_img, left_fit, right_fit):
    ploty = np.linspace(0, wrap_img.shape[0]-1, wrap_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty

def getCurveRadius(ploty, x, imgSizeY, xm_per_pix, ym_per_pix):
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, x*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval1 = np.max(ploty)-20
    curverad1 = ((1 + (2*fit_cr[0]*y_eval1*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    y_eval2 = np.max(ploty)-60
    curverad2 = ((1 + (2*fit_cr[0]*y_eval2*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    y_eval3 = np.max(ploty)-100
    curverad3 = ((1 + (2*fit_cr[0]*y_eval3*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    curverad = (curverad1 + curverad2 + curverad3) / 3
    return curverad

def getCarPositionOffCenter(left_fit, right_fit, img_size_x, img_size_y, xm_per_pix):
    base_left = left_fit[0]*img_size_y**2 + left_fit[1]*img_size_y + left_fit[2]
    base_right = right_fit[0]*img_size_y**2 + right_fit[1]*img_size_y + right_fit[2]
    car_pos = img_size_x / 2.
    centerOfLanes = base_left+((base_right-base_left)/2)
    offset = (centerOfLanes - car_pos)*xm_per_pix
    return offset

def project_back(wrap_img, origin_img, left_fitx, right_fitx, ploty, M, left_curverad, right_curverad, car_offset):
    warp_zero = np.zeros_like(wrap_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, M)
    # Combine the result with the original image
    result = cv2.addWeighted(origin_img, 1, newwarp, 0.3, 0)
    cv2.putText(result, 'left curvature: ' + str(round(left_curverad, 1)) + ' m', (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, 'right curvature: ' + str(round(right_curverad, 1)) + ' m', (50, 80), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, 'offset: ' + str(round(car_offset * 100., 1)) + ' cm', (50, 110), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    return result



with open('calibration_paraeters.pkl', mode='rb') as f:
    dist_pickle = pickle.load(f)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

'''
the image should be RGB
'''
def process_image(image):
    undistorted = cal_undistort(image, mtx, dist)
    binary_img = combine_filters(undistorted)

    wrap_offset = 150
    src_corners = [(603, 445), (677, 445), (1105, binary_img.shape[0]), (205, binary_img.shape[0])]
    dst_corners = [(205 + wrap_offset, 0), (1105 - wrap_offset, 0), (1105 - wrap_offset, binary_img.shape[0]),
                   (205 + wrap_offset, binary_img.shape[0])]

    M = cv2.getPerspectiveTransform(np.float32(src_corners), np.float32(dst_corners))
    M_inverse = cv2.getPerspectiveTransform(np.float32(dst_corners), np.float32(src_corners))

    wrap_img = perspective_transform(binary_img, M)

    left_fit, right_fit, out_img = find_line_fit(wrap_img)
    left_fitx, right_fitx, ploty = get_fit_xy(wrap_img, left_fit, right_fit)

    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_curverad = getCurveRadius(ploty, left_fitx, wrap_img.shape[0], xm_per_pix, ym_per_pix)
    right_curverad = getCurveRadius(ploty, right_fitx, wrap_img.shape[0], xm_per_pix, ym_per_pix)

    car_offset = getCarPositionOffCenter(left_fit, right_fit, wrap_img.shape[1], wrap_img.shape[0], xm_per_pix)
    result = project_back(wrap_img, undistorted, left_fitx, right_fitx, ploty, M_inverse, left_curverad, right_curverad,
                          car_offset)
    return result


import os

images = os.listdir('test_images')
fig = plt.figure(figsize=(16, 36))
for i, fname in enumerate(images):
    img = cv2.imread(os.path.join('test_images', fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = process_image(img)
    plt.imsave(os.path.join('output_images', fname), result)
    plt.subplot(4, 2, i + 1)
    plt.imshow(result)
plt.show()
