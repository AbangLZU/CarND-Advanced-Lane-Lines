from __future__ import print_function

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
from moviepy.editor import VideoFileClip
from IPython.display import HTML
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


def find_line_pixel_from_detected(wrap_img, c, margin=70):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = wrap_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    lane_inds = ((nonzerox > (c[0] * (nonzeroy ** 2) + c[1] * nonzeroy + c[2] - margin)) &
                 (nonzerox < (c[0] * (nonzeroy ** 2) + c[1] * nonzeroy + c[2] + margin)))
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    return x, y


def find_line_pixel_without_detected(wrap_img, flag='L', nwindows=9, margin=100, minpix=50):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = wrap_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(wrap_img[wrap_img.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point
    midpoint = np.int(histogram.shape[0] / 2)
    if flag == 'L':
        base = np.argmax(histogram[:midpoint])
    else:
        base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows
    window_height = np.int(wrap_img.shape[0] / nwindows)
    # Current positions to be updated for each window
    current = base
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = wrap_img.shape[0] - (window + 1) * window_height
        win_y_high = wrap_img.shape[0] - window * window_height

        win_x_low = current - margin
        win_x_high = current + margin

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            current = np.int(np.mean(nonzerox[good_inds]))
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    return x, y


# Generate x and y values for plotting
def get_fit_xy(wrap_img, left_fit, right_fit):
    ploty = np.linspace(0, wrap_img.shape[0]-1, wrap_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


def getCurveRadius(ploty, x, xm_per_pix, ym_per_pix):
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


def project_back(wrap_img, origin_img, left_fitx, right_fitx, ploty, M):
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
    return result

#########


# Define a class to receive the characteristics of each line detection
class Line(object):
    def __init__(self, buffer_len=5):
        # the buffer length
        self.buffer_len = buffer_len
        # currunt number of buffered data
        self.num_buffered = 0
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients over the last n iterations
        self.last_n_coeffs = deque([], maxlen=buffer_len)
        # last n x values
        self.last_n_x_values = deque([], maxlen=buffer_len)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # current fit y values
        self.fit_y_value = np.linspace(0, 719, 720)
        # current fit x values
        self.fit_x_value = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # the line x at bottom
        self.line_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # the average distance of left and right line
        self.line_distance = None

    def set_allxy(self, x, y):
        self.allx = x
        self.ally = y

    def set_current_fit(self):
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

    def set_fit_x(self):
        self.fit_x_value = self.current_fit[0] * self.fit_y_value ** 2 + self.current_fit[1] * self.fit_y_value \
                           + self.current_fit[2]

    def get_diffs(self):
        if self.num_buffered > 0:
            self.diffs = self.current_fit - self.best_fit
        else:
            self.diffs = np.array([0, 0, 0], dtype='float')

    def set_curvature_pos(self):
        self.radius_of_curvature = getCurveRadius(self.fit_y_value, self.fit_x_value, 3.7 / 700, 30. / 720)
        y_eval = max(self.fit_y_value)
        self.line_pos = self.current_fit[0] * y_eval ** 2 \
                        + self.current_fit[1] * y_eval \
                        + self.current_fit[2]
        basepos = 640
        self.line_base_pos = (self.line_pos - basepos) * 3.7 / 700.0

    # here come sanity checks of the computed metrics
    def sanity_check(self):
        is_pass = True
        if abs(self.line_base_pos) > 2.8:
            print('lane too far away')
            is_pass = False
        if self.num_buffered > 0:
            relative_delta = self.diffs / self.best_fit
            # allow maximally this percentage of variation in the fit coefficients from frame to frame
            if not (abs(relative_delta) < np.array([0.7, 0.5, 0.15])).all():
                print('fit coeffs too far off [%]', relative_delta)
                is_pass = False
        return is_pass

    def set_avgx(self):
        fits = self.last_n_x_values
        if len(fits) > 0:
            avg = 0
            for fit in fits:
                avg += np.array(fit)
            avg = avg / len(fits)
            self.bestx = avg


    def set_avgcoeffs(self):
        coeffs = self.last_n_coeffs
        if len(coeffs)>0:
            avg=0
            for coeff in coeffs:
                avg +=np.array(coeff)
            avg = avg / len(coeffs)
            self.best_fit = avg

    def add_data(self):
        self.last_n_x_values.append(self.fit_x_value)
        self.last_n_coeffs.append(self.current_fit)
        assert len(self.last_n_x_values) == len(self.last_n_coeffs)
        self.num_buffered = len(self.last_n_x_values)

    def pop_data(self):
        if self.num_buffered > 0:
            self.last_n_x_values.popleft()
            self.last_n_coeffs.popleft()
            assert len(self.last_n_x_values) == len(self.last_n_coeffs)
            self.num_buffered = len(self.last_n_x_values)

    def update_para(self, x, y):
        self.set_allxy(x, y)
        self.set_current_fit()
        self.set_fit_x()
        self.set_curvature_pos()
        self.get_diffs()
        if self.sanity_check():
            self.detected = True
            self.add_data()
            self.set_avgx()
            self.set_avgcoeffs()
        else:
            self.detected = False
            self.pop_data()
            if self.num_buffered > 0:
                self.set_avgx()
                self.set_avgcoeffs()


class ImageInfo(object):
    def __init__(self, raw_image):
        self.img_size = raw_image.shape
        self.wrap_offset = 100
        src_corners = [(603, 445), (677, 445), (1105, self.img_size[0]), (205, self.img_size[0])]
        dst_corners = [(205 + self.wrap_offset, 0), (1105 - self.wrap_offset, 0), (1105 - self.wrap_offset, self.img_size[0]),
                       (205 + self.wrap_offset, self.img_size[0])]
        self.M = cv2.getPerspectiveTransform(np.float32(src_corners), np.float32(dst_corners))
        self.M_inverse = cv2.getPerspectiveTransform(np.float32(dst_corners), np.float32(src_corners))


def makeTextDataImage(warped_bin, lineLeft, lineRight, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1):
    warpedImgSize = (warped_bin.shape[1], warped_bin.shape[0])
    imgSize = (390, 500, 3)
    fontColor = (0, 255, 0)
    data_img = np.zeros(imgSize, dtype=np.uint8)
    # calculate the CurveRadius and CarOffset by using the bestx
    if lineLeft.bestx != None:
        curvature = getCurveRadius(lineLeft.fit_y_value, lineLeft.bestx, 3.7 / 700, 30. / 720)
        cv2.putText(data_img, 'left crv rad: {:.0f}m'.format(curvature), (10, 30), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'l coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineLeft.best_fit[0], lineLeft.best_fit[1], lineLeft.best_fit[2]), (10, 110), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'l differs: {:.7f}, {:.2f}, {:.2f}'.format(lineLeft.diffs[0], lineLeft.diffs[1], lineLeft.diffs[2]), (10, 160), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
    if lineRight.bestx != None:
        curvature = getCurveRadius(lineRight.fit_y_value, lineRight.bestx, 3.7 / 700, 30. / 720)
        cv2.putText(data_img, 'right crv rad: {:.0f}m'.format(curvature), (10, 60), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'r coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineRight.best_fit[0], lineRight.best_fit[1], lineRight.best_fit[2]), (10, 140), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'r differs: {:.7f}, {:.2f}, {:.2f}'.format(lineRight.diffs[0], lineRight.diffs[1], lineRight.diffs[2]), (10, 190), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
    if lineLeft.best_fit != None and lineRight.best_fit != None:
        car_offset = getCarPositionOffCenter(left.best_fit, right.best_fit, warpedImgSize[0], warpedImgSize[1],
                                             3.7 / 700)
        cv2.putText(data_img, 'off center: {:.1f}m'.format(car_offset), (10, 280), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
    return data_img


def makeCtrlImg(finalImg, textDataImg, diagImg, warped_bin):
    imgSize = (750, 1280, 3)
    ctrl_img = np.zeros(imgSize, dtype=np.uint8)
    #ctrl_img = ctrl_img + (30,30,30)
    smallFinal = cv2.resize(finalImg, (0, 0), fx=0.5, fy=0.5)
    smallFinalSize = (smallFinal.shape[1], smallFinal.shape[0])
    ctrl_img[0:smallFinalSize[1], 0:smallFinalSize[0]] = smallFinal

    xOffset = smallFinalSize[0]+20
    yOffset = 35
    smallWarped = cv2.resize(warped_bin, (0, 0), fx=0.45, fy=0.45)
    smallWarpedSize = (smallWarped.shape[1], smallWarped.shape[0])
    ctrl_img[yOffset:(yOffset+smallWarpedSize[1]), xOffset:(xOffset+smallWarpedSize[0])] = smallWarped

    xOffset = smallFinalSize[0]+20
    yOffset = smallWarpedSize[1] + 35
    smallDiag = cv2.resize(diagImg, (0, 0), fx=0.45, fy=0.45)
    smallDiagSize = (smallDiag.shape[1], smallDiag.shape[0])
    ctrl_img[yOffset:(yOffset+smallDiagSize[1]), xOffset:(xOffset+smallDiagSize[0])] = smallDiag

    yOffset = smallFinalSize[1]
    #smallDiag = cv2.resize(textDataImg, (0,0), fx=0.5, fy=0.5)
    smallTextSize = (textDataImg.shape[1], textDataImg.shape[0])
    ctrl_img[yOffset:(yOffset+smallTextSize[1]), 0:smallTextSize[0]] = textDataImg
    return ctrl_img


'''
the image should be RGB
'''
def process_image(image):
    global image_info
    global left
    global right

    if image_info is None:
        image_info = ImageInfo(image)
    undistorted = cal_undistort(image, mtx, dist)
    binary_img = combine_filters(undistorted)

    wrap_img = perspective_transform(binary_img, image_info.M)

    if left.detected:
        l_x, l_y = find_line_pixel_from_detected(wrap_img, left.best_fit)
    else:
        l_x, l_y = find_line_pixel_without_detected(wrap_img, 'L')
    left.update_para(l_x, l_y)

    if right.detected:
        r_x, r_y = find_line_pixel_from_detected(wrap_img, right.best_fit)
    else:
        r_x, r_y = find_line_pixel_without_detected(wrap_img, 'R')
    right.update_para(r_x, r_y)

    out_img_1 = np.dstack((wrap_img, wrap_img, wrap_img)) * 255
    out_img_1[l_y, l_x] = [255, 0, 0]
    out_img_1[r_y, r_x] = [0, 0, 255]

    out_img_2 = np.zeros_like(out_img_1)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left.bestx - 10, left.fit_y_value]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left.bestx + 10,
                                                                        left.fit_y_value])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right.bestx - 10, right.fit_y_value]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right.bestx + 10,
                                                                         right.fit_y_value])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img_2, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(out_img_2, np.int_([right_line_pts]), (0, 255, 0))
    result = project_back(wrap_img, undistorted, left.bestx, right.bestx, left.fit_y_value, image_info.M_inverse)

    text_image = makeTextDataImage(wrap_img, left, right)
    control_img = makeCtrlImg(result, text_image, out_img_2, out_img_1)
    return control_img

with open('calibration_paraeters.pkl', mode='rb') as f:
    dist_pickle = pickle.load(f)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

## global part
left = Line(buffer_len=7)
right = Line(buffer_len=10)
image_info = None

output_dir = 'test_videos_output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
video_file = 'project_video.mp4'
clip1 = VideoFileClip(video_file)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(os.path.join(output_dir, video_file), audio=False)