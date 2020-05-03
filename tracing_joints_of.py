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

# use more algo like scene detection to match kernel instead of matching point
# calculate dense optical flow around box of confirmed feature
#


def get_next_frame_joints(prev_frame, curr_frame, joint_list):
    '''
    Inputs: curr_frame - the current frame of the video
            joint_list - an n X 2 array, where n is the number of joints and holds the xy coords of each joint
    outputs: a list of joints for the current frame
    '''

    # go through joint_list of previous frame, crop image to size around each joint, put in get_interest_point, grab
    # new coords, put them in joint_coords of current frame, return list of joints.
    lk_params = dict(winSize=(50, 50),
                     maxLevel=3)
    old_pts = np.float32(joint_list)
    np.reshape(old_pts, (-1, 1, 2))
    next_joints, stat, err = cv2.calcOpticalFlowPyrLK(
        prev_frame, curr_frame, old_pts, None, **lk_params)

    bad_new = stat == 0
    bad_new = np.hstack((bad_new, bad_new))
    next_joints[bad_new] = joint_list[bad_new]
    return np.asarray(next_joints)


def trace_joints(video, joint_list):
    '''
    input: video - the video that we're are tracing forward
           joint_list - the list of coordinates of joints for the first frame
    output: a list of jointlists for the whole video

    '''
    video_joint_list = []
    # takes out the first frame
    print("Taking out first frame.")
    ret, old_frame = video.read()
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    show_frame(old_frame, joint_list)

    # for each frame, it adds the joint list of the previous frame into a list and then
    # uses interest_points to find the joint list of the current frame, and repeats
    ret, new_frame = video.read()
    i = 2
    while ret:
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        if i % 500 == 0:
            show_frame(new_frame, joint_list)
        i += 1
        new_joints = joint_list
        video_joint_list.append(new_joints)
        joint_list = get_next_frame_joints(old_frame, new_frame, joint_list)
        old_frame = new_frame
        ret, new_frame = video.read()
    return np.asarray(video_joint_list)


def show_frame(bw_frame, points):
    plt.imshow(bw_frame)
    plt.scatter(x=points[:, 0], y=points[:, 1], c='r', s=40)
    plt.show()


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
    plt.gray()
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