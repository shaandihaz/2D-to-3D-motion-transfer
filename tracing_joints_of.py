import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import convolve, sobel
import cv2
import json


def bound(joint_coords, image):
    old_joints = joint_coords
    np.clip(joint_coords[:, 0], a_min=0,
            a_max=image.shape[1]-1, out=joint_coords[:, 0])
    np.clip(joint_coords[:, 1], a_min=0,
            a_max=image.shape[0]-1, out=joint_coords[:, 1])
    return joint_coords

# calculate dense optical flow around box of confirmed feature


def est_next_points(old_points, curr_points):
    scale = 0.05
    diff = curr_points - old_points
    return curr_points + scale * diff


def get_next_frame_joints_lk(par_frame, curr_frame, par_joints, grand_joints):
    '''
    Inputs: curr_frame - the current frame of the video
            joint_list - an n X 2 array, where n is the number of joints and holds the xy coords of each joint
    outputs: a list of joints for the current frame
    '''

    lk_params = dict(winSize=(50, 50),
                     maxLevel=3)
    old_pts = np.float32(par_joints)
    np.reshape(old_pts, (-1, 1, 2))
    est = est_next_points(grand_joints, par_joints)
    next_joints, stat, err = cv2.calcOpticalFlowPyrLK(
        par_frame, curr_frame, old_pts, est, **lk_params)

    for i in range(stat.shape[0]):
        if stat[i] == 0:
            next_joints[i] = est[i]
    return bound(np.asarray(next_joints), curr_frame)


def trace_joints(video, curr_joints):
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
    show_frame(old_frame, curr_joints, 1)

    # for each frame, it adds the joint list of the previous frame into a list and then
    # uses interest_points to find the joint list of the current frame, and repeats
    ret, new_frame = video.read()
    old_joints = curr_joints
    i = 2
    while ret:
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        if i >= 150:
            show_frame(new_frame, curr_joints, i)
        i += 1
        new_joints = curr_joints
        video_joint_list.append(new_joints)
        next_joints = get_next_frame_joints_lk(
            old_frame, new_frame, curr_joints, old_joints)
        old_joints = curr_joints
        curr_joints = next_joints
        old_frame = new_frame
        ret, new_frame = video.read()
    return np.asarray(video_joint_list)


def show_frame(bw_frame, points, i):
    plt.imshow(bw_frame)

    plt.scatter(x=points[:, 0], y=points[:, 1], c=['#397916', '#8C164F', '#5F8EEB', '#CA505D', '#9B4196', '#612006',
                                                   '#9AFAC4', '#CF91E1', '#A68875', '#5F3881', '#837FE0', '#D9AFB4', '#C19AE7', '#4EF727', '#00A140'], s=40)
    plt.savefig("Img{}".format(i))
    plt.close()
    # plt.show()


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
