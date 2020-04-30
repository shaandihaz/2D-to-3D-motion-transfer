import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import convolve, sobel
import cv2
import json


def get_interest_point(image, feature_width):

    # this is essentially the get_interest_points function from project 2, just revised so that
    # it only grabs 1 feature point
    '''
    :params:
    :image: the cropped frame around the specific joint we're looking for 
    :feature_width: don't do anything with it yet, may be useful later

    :returns:
    :x: x coordinate of the interest point in the image
    :y: y coordinate of the interest point in the image

    '''

    # yDerivKer = np.array([[-2, -1, 0, 1, 2],[-2,-1,0,1,2], [-2,-1,0,1,2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
    # xDerivKer = np.array([[2,2,2,2,2], [1,1,1,1,1],[0,0,0,0,0],[-1,-1,-1,-1,-1], [-2,-2,-2,-2,-2]])
    # yDer = convolve(image, yDerivKer)
    # xDer = convolve(image, xDerivKer)

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
    arr = feature.peak_local_max(C, num_peaks=1)
    x = arr[:, 1][0]
    y = arr[:, 0][0]

    return x, y


def get_next_frame_joints(curr_frame, joint_list):
    '''
    Inputs: curr_frame - the current frame of the video
            joint_list - an n X 2 array, where n is the number of joints and holds the xy coords of each joint
    outputs: a list of joints for the current frame
    '''

    # go through joint_list of previous frame, crop image to size around each joint, put in get_interest_point, grab
    # new coords, put them in joint_coords of current frame, return list of joints.
    joint_coords = []
    for joint in joint_list:
        x, y = get_interest_point(
            # grab most significant 2D point in the 64x64 subimage around joint
            curr_frame[joint[1] - 32:joint[1] + 32, joint[0] - 32:joint[0] + 32], 16)
        new_joint = [x + joint[0] - 32, y + joint[1] - 32]
        joint_coords.append(new_joint)
    return np.asarray(joint_coords)


def trace_joints(video, joint_list):
    '''
    input: video - the video that we're are tracing forward
           joint_list - the list of coordinates of joints for the first frame
    output: a list of jointlists for the whole video

    '''
    video_joint_list = []
    # takes out the first frame
    print("Taking out first frame.")
    ret, frame = video.read()
    # for each frame, it adds the joint list of the previous frame into a list and then
    # uses interest_points to find the joint list of the current frame, and repeats
    ret, frame = video.read()
    i = 2
    while ret:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print("Frame {}".format(i))
        if i % 10 == 0:

        i += 1
        print(joint_list)
        new_joints = joint_list
        video_joint_list.append(new_joints)
        joint_list = get_next_frame_joints(frame, joint_list)
        ret, frame = video.read()
    return np.asarray(video_joint_list)


def show_frame(bw_frame, points):
    plt.imshow(bw_frame)
    plt.


def read_video(v_name):
    '''
    returns an iterable of the images in a video
    input: v_name: the name of the video to process
    output: a cv2.VideoCapture object (essentially an iterable over the images in the video)
    '''
    vid = cv2.VideoCapture(v_name)
    return vid


def main():
    print("Getting video.")
    vid = read_video('../hd00_03.mp4')
    # Make sure joint indices are integers
    first_joints = np.asarray([
        [1077, 930],
        [1097, 779],
        [1101, 651],
        [1182, 946],
        [1181, 780],
        [1198, 659],
        [1149, 629],
        [1058, 628],
        [1225, 636],
        [1070, 540],
        [1236, 558],
        [1083, 431],
        [1216, 424],
        [1153, 322],
        [1157, 390],
    ])
    res = trace_joints(vid, first_joints)
    print(res)


if __name__ == '__main__':
    main()
