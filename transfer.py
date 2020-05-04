import numpy as np





# Finds the next set of 3d points given an initial frame with 2d and 3d points,
# the next set of 2d points, the camera projection matrix, and the graph describing
# the fixed distances between the points
def find_next(points_2d_a, points_2d_b, points_3d_a, proj_mat, graph)
    # Not sure if this is the best option, but trying it for now, this is the one we use to assume our starting point
    closest_ind = find_closest_ind(points_2d_a, points_2d_b)
    # Find all the rays
    rays = points_to_rays(points_2d_b, proj_mat)
    # find the closest point for the starting point we selected
    closest = closest_on_ray(points_3d_a[closest_ind], rays[closest_ind])
    done = {}
    toDo = {closest_ind : (graph[closest_ind], closest)}
    while len(toDo) > 0:
        # Get an arbitrary node
        ind = toDo.keys()[0]
        node = toDo.pop(ind)
        done[ind] = node
        # Iterate through the neighbors to find their estimated 3D positions, then add them to todo
        for neighbor in node[0].neighbors:
            if not (neighbor[0].ind in done or neighbor[0].ind in toDo):
                toDo[neighbor.ind] = (neighbor[0], find_next_on_ray(node[1], points_3d_a[neighbor.ind], rays[neighbor.ind], neighbor[1]))
    points_3d_b = np.zeros(points_3d_a.shape)
    for ind in done:
        points_3d_b[ind] = done[ind][1]
    
    return points_3d_b

# Find the closest point on a ray to a given 3d point 
def closest_on_ray(point_3d, ray):
    v1 = point_3d - ray.o
    hyp = np.linalg.norm(v1)
    theta = np.arccos(np.dot(ray.dir, v1)/ (hyp * np.linalg.norm(ray.dir)))
    t = np.cos(theta) * hyp
    return (t * ray.dir) + ray.o

# This function will find the index of the pair of points in these
# sets that are closest to each other in euclidean distance
def find_closest_ind(points_2d_a, points_2d_b):
    return np.argmin(np.sum(np.square(points_2d_a - points_2d_b), axis=1))
    pass

# This function will find the point that lies on a given ray that has
# a specified distance away from a given point and is the closest
def find_next_on_ray(point_3d, point_3d_old, ray, dist):
    closest = closest_on_ray(point3d, ray)
    v1 = closest - point_3d
    closest_dist = np.linalg.norm(v1)
    if closest_dist >= dist:
        return closest
        #return ((v1 / closest_dist) * dist) + point_3d
    else:
        t = np.sin(np.arccos(closest_dist/dist)) * dist
        a = closest + (ray.dir * t)
        b = closest - (ray.dir * t)
        if np.sum(np.square(a - point_3d_old)) > np.sum(np.square(b - point_3d_old)):
            return b
        else:
            return a
    pass

# This function will take in 2d points and the intrinsic matrix and get 3D rays
# that pass through each 2d point
def points_to_rays(points_2d, K):
    # Note: ray.dir must be normalized
    
    
    return 

