import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import convolve, sobel
import cv2
import json
import os
import transfer


def bound(joint_coords, image):
    """
    Ensure joint coordinates are in the image using np.clip.
    """
    old_joints = joint_coords
    np.clip(joint_coords[:, 0], a_min=0,
            a_max=image.shape[1]-1, out=joint_coords[:, 0])
    np.clip(joint_coords[:, 1], a_min=0,
            a_max=image.shape[0]-1, out=joint_coords[:, 1])
    return joint_coords

# calculate dense optical flow around box of confirmed feature


def est_next_points(old_points, curr_points):
    """
    Estimates next points by continuing the motion from last two frames.
    """
    scale = 0.05
    # motion from grandparent frame to parent frame.
    diff = curr_points - old_points
    return curr_points + scale * diff


def get_next_frame_joints_sparse(par_frame, curr_frame, par_joints, grand_joints):
    '''
    Given two consecutive frames with the joints of the last two frames, finds the joint coordinates of the next frame.
    Inputs: 
            par_frame    - the previous frame of the video
            curr_frame   - the current frame of the video
            par_joints   - the joint coordinates in par_frame
            grand_joints - the joint coordinates in the frame before par_frame
    outputs: the estimated joint coordinates in curr_frame
    '''

    # formats inputs for calcOpticalFlow
    lk_params = dict(winSize=(50, 50),  # window size of each pyrimid level
                     maxLevel=3)  # number of pyramid levels

    # old_pts must be a float32 array of size (num_joints) x 1 x 2
    old_pts = np.float32(par_joints)
    np.reshape(old_pts, (-1, 1, 2))

    # first estimate of curr_frame's joint coords
    est = est_next_points(grand_joints, par_joints)

    # estimates next points using optical flow
    next_joints, stat, err = cv2.calcOpticalFlowPyrLK(
        par_frame, curr_frame, old_pts, est, **lk_params)

    # if stat[i] = 0, then our estimate is wrong, so we just make it our initial estimate.
    for i in range(stat.shape[0]):
        if stat[i] == 0:
            next_joints[i] = est[i]

    # joints must be within image
    return bound(np.asarray(next_joints), curr_frame)


def trace_joints(video, curr_joints, show2D=False, save2D=False, show3D=False, save3D=False):
    '''
    Calculates the joint locations throughout the video using the joint locations at frame 0.
    input: video - the video that we're are tracing forward
           joint_list - the list of coordinates of joints for the first frame
    output: a list of jointlists for the whole video

    '''
    video_joint_list = []

    print("Taking out first frame.")
    ret, old_frame = video.read()
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # bw image
    visualize_2d(old_frame, curr_joints, 1, show2D, save2D)
    visualize_3d(curr_joints, 1, show3D, save3D)

    # for each frame, it adds the joint list of the previous frame into a list and then
    # uses interest_points to find the joint list of the current frame, and repeats
    ret, new_frame = video.read()
    old_joints = curr_joints
    i = 2
    while ret:
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        if i >= 150 and i % 10 == 0:
            visualize_2d(new_frame, curr_joints, i, show2D, save2D)
            visualize_3d(curr_joints, i, show3D, save3D)
        i += 1
        new_joints = curr_joints
        video_joint_list.append(new_joints)
        next_joints = get_next_frame_joints_sparse(
            old_frame, new_frame, curr_joints, old_joints)
        old_joints = curr_joints
        curr_joints = next_joints
        old_frame = new_frame
        ret, new_frame = video.read()
    return np.asarray(video_joint_list)


points3 = None
points_prev = None
graph = None


def visualize_3d(points, i, show3D, save3D):
    if not show3D and not save3D:
        return
    global points3
    global points_prev
    global graph
    if points3 is None:
        points3, graph = transfer.find_initial(points)
    else:
        points3 = transfer.find_next(points_prev, points, points3, graph)
    points_prev = np.copy(points)
    ax = plt.axes(projection='3d')
    ax.axes.set_xlim3d(0, 2000)
    ax.axes.set_ylim3d(0, 1100)
    ax.axes.set_zlim3d(0, 2200)
    ax.scatter3D(xs=points3[:, 0], zs=np.subtract(2000.0, points3[:, 1]), ys=points3[:, 2], c=['#397916', '#8C164F', '#5F8EEB', '#CA505D', '#9B4196', '#612006',
                                                                                               '#9AFAC4', '#CF91E1', '#A68875', '#5F3881', '#837FE0', '#D9AFB4', '#C19AE7', '#4EF727', '#00A140'], s=40)
    ax.azim = -85
    ax.elev = 10
    if show3D:
        plt.show()
    else:
        plt.savefig("3dImg{}".format(i))
        plt.close()


def visualize_2d(bw_frame, points, i, show2D, save2D):
    if not show2D and not save2D:
        return
    plt.imshow(bw_frame)

    plt.scatter(x=points[:, 0], y=points[:, 1], c=['#397916', '#8C164F', '#5F8EEB', '#CA505D', '#9B4196', '#612006',
                                                   '#9AFAC4', '#CF91E1', '#A68875', '#5F3881', '#837FE0', '#D9AFB4', '#C19AE7', '#4EF727', '#00A140'], s=40)
    if show2D:
        plt.show()
    else:
        plt.savefig("Img{}".format(i))
        plt.close()


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
    print(os.getcwd())
    vid = read_video('../hd_00_03.mp4')
    print(vid)
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
    res = trace_joints(vid, first_joints, save=False, show=False)
    print(res)


if __name__ == '__main__':
    main()
