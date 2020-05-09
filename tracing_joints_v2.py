import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import convolve, sobel
from scipy.spatial.distance import cdist
import cv2


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image


    '''

    # These are placeholders - replace with the coordinates of your interest points!
    xs = np.zeros(1)
    ys = np.zeros(1)

    yDer = sobel(image, 0)
    xDer = sobel(image, 1)
    gaussYDer2 = filters.gaussian(yDer * yDer)
    gaussXDer2 = filters.gaussian(xDer * xDer)
    gausXYder = filters.gaussian(xDer * yDer)
    addXY2 = gaussXDer2 + gaussYDer2
    C = gaussXDer2 * gaussYDer2 - \
        (gausXYder*gausXYder) - 0.02*(addXY2 * addXY2)
    threshold_num = 0.001  # need actual number
    C[C < threshold_num] = 0.0
    arr = feature.peak_local_max(
        C, feature_width, exclude_border=feature_width//2)
    xs = arr[:, 1]
    ys = arr[:, 0]

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''
    width = feature_width // 4
    xDer = sobel(image, 1)
    yDer = sobel(image, 0)
    grad_mag = np.sqrt(np.square(xDer) + np.square(yDer))
    grad_ori = np.arctan2(yDer, xDer) + np.pi
    features = []
    inc = np.pi/4
    bins = np.array([0, inc, 2*inc, 3*inc, 4*inc, 5 *
                     inc, 6*inc, 7*inc, 8*inc + 0.001])
    for z in range(len(x)):
        xs = int(x[z])
        ys = int(y[z])
        xlb = xs - width*2
        xhb = xs + width*2
        ylb = ys - width*2
        yhb = ys + width*2
        descriptor = []
        if (xlb < 0 or ylb < 0 or xhb >= image.shape[1] or yhb >= image.shape[0]):

            continue
        else:
            for i in range(xlb, xhb, width):
                for j in range(ylb, yhb, width):
                    sublist = [0] * 8
                    submag = grad_mag[j:j + width, i:i + width]
                    subori = grad_ori[j:j + width, i:i + width]
                    digits = np.digitize(subori, bins)
                    for k in range(width):
                        for l in range(width):
                            index = digits[k, l] - 1
                            sublist[index] += submag[k][l]
                    descriptor.extend(sublist)
            norm = np.linalg.norm(descriptor)
            descriptor = descriptor / norm
            descriptor[descriptor > 0.2] = 0.2
            norm = np.linalg.norm(descriptor)
            descriptor = descriptor / norm
            descriptor = [xs, ys] + descriptor
            features.append(descriptor)
    #features = np.asarray(features)
    return features

def match_features(joint_features, frame_features):
    new_joints = []
    for joint in joint_features:
        filtered_frame_feat = []
        frame_points = []
        for feat in frame_features:
            if joint[0] - 8 <= frame_features[0] <= joint[0] + 8 and joint[1] - 8 <= frame_features[1] <= joint[1] + 8:
                frame_points.append([feat.pop(0), feat.pop(0)])
                filtered_frame_feat.append(feat)
        arr_joint = np.asarray(joint)
        arr_frame_feat = np.asarray(filtered_frame_feat)
        dist_arr = cdist(arr_joint, arr_frame_feat)
        new_joint = frame_points[np.argsort(dist_arr)[0]]
        new_joints.append(new_joint)
    return new_joints

        
        

    


def get_next_frame_joints(curr_frame, joint_features):
    '''
    Inputs: curr_frame - the current frame of the video
            joint_list - an n X 2 array, where n is the number of joints and holds the xy coords of each joint
    outputs: a list of joints for the current frame
    '''

    # go through joint_list of previous frame, crop image to size around each joint, put in get_interest_point, grab
    # new coords, put them in joint_coords of current frame, return list of joints.

    '''
    #this is leftover code from the other version
    for joint in joint_list:
        x, y = get_interest_point(
            # grab most significant 2D point in the 64x64 subimage around joint
            curr_frame[joint[1] - 32:joint[1] + 32, joint[0] - 32:joint[0] + 32], 16)
        new_joint = [x,y]
        joint_coords.append(new_joint)
    '''
    (x, y) = get_interest_points(curr_frame, 16)
    frame_feat = get_features(curr_frame, x, y, 16)

    return match_features(joint_features, frame_feat)


def trace_joints(video, joint_list):
    '''
    input: video - the video that we're are tracing forward
           joint_list - the list of coordinates of joints for the first frame
    output: a list of jointlists for the whole video

    '''
    video_joint_list = []

    # for each frame, it adds the joint list of the previous frame into a list and then
    # uses interest_points to find the joint list of the current frame, and repeats
    ret, first_frame = video.read()
    if not ret:
        print("Failure to load video")
        return None
    first_features = get_features(
        first_frame, joint_list[:, 1], joint_list[:, 0], 16)
    # for each frame, it adds the joint list of the previous frame into a list and then
    # uses interest_points to find the joint list of the current frame, and repeats
    ret, frame = video.read()
    while ret:
        new_joints = joint_list
        video_joint_list.append(new_joints)
        joint_list = get_next_frame_joints(frame, first_features)
        first_features = get_features(frame, joint_list[:, 1], joint_list[:, 0], 16)
        ret, frame = video.read()
    return np.asarray(video_joint_list)

def read_video(v_name):
    '''
    returns an iterable of the images in a video
    input: v_name: the name of the video to process
    output: a cv2.VideoCapture object (essentially an iterable over the images in the video)
    '''
    vid = cv2.VideoCapture(v_name)
    return vid
