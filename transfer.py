import numpy as np


class Ray:
    def __init__(self, o, direction):
        self.dir = direction
        self.o = o

class GraphNode:
    def __init__(self, ind):
        self.neighbors = []
        self.ind = ind

    def add_neighbor(self, neighbor, points, recur):
        self.neighbors.append((neighbor, np.linalg.norm(points[neighbor.ind] - points[self.ind])))
        if recur:
            neighbor.add_neighbor(self, points, False)

def build_graph(points_2d):
    graph = []
    for i in range(points_2d.shape[0]):
        graph.append(GraphNode(i))
    graph[13].add_neighbor(graph[14], points_2d, True)
    graph[14].add_neighbor(graph[12], points_2d, True)
    graph[14].add_neighbor(graph[9], points_2d, True)
    graph[12].add_neighbor(graph[10], points_2d, True)
    graph[10].add_neighbor(graph[8], points_2d, True)
    graph[11].add_neighbor(graph[9], points_2d, True)
    graph[7].add_neighbor(graph[11], points_2d, True)
    graph[6].add_neighbor(graph[14], points_2d, True)
    graph[6].add_neighbor(graph[2], points_2d, True)
    graph[6].add_neighbor(graph[5], points_2d, True)
    graph[2].add_neighbor(graph[1], points_2d, True)
    graph[1].add_neighbor(graph[0], points_2d, True)
    graph[5].add_neighbor(graph[4], points_2d, True)
    graph[4].add_neighbor(graph[3], points_2d, True)
    return graph

    


# Main loop:
#   find 2d points init
#   call find_initial
#   find next set of 2d points
#   repeat :
#       call find_next
#       find next set of 2d points
#
#
def find_initial(points_2d):
    graph = build_graph(points_2d)
    _, points = points_to_rays(points_2d)
    return points, graph


# Finds the next set of 3d points given an initial frame with 2d and 3d points,
# the next set of 2d points, the camera projection matrix, and the graph describing
# the fixed distances between the points
def find_next(points_2d_a, points_2d_b, points_3d_a, graph):
    # Not sure if this is the best option, but trying it for now, this is the one we use to assume our starting point
    closest_ind = find_closest_ind(points_2d_a, points_2d_b)
    # Find all the rays
    rays, _ = points_to_rays(points_2d_b)
    # find the closest point for the starting point we selected
    closest = closest_on_ray(points_3d_a[closest_ind], rays[closest_ind])
    done = {}
    toDo = {closest_ind : (graph[closest_ind], closest)}
    attempts = 0
    while len(toDo) > 0:
        # Get an arbitrary node
        ind = list(toDo.keys())[0]
        node = toDo.pop(ind)
        done[ind] = node
        #attempt = False
        # Iterate through the neighbors to find their estimated 3D positions, then add them to todo
        for neighbor in node[0].neighbors:
            if not (neighbor[0].ind in done or neighbor[0].ind in toDo):
                step_back, tmp_point = find_next_on_ray(node[1], points_3d_a[neighbor[0].ind], rays[neighbor[0].ind], neighbor[1])
                if step_back is None:
                    toDo[neighbor[0].ind] = (neighbor[0], tmp_point)
                else:
                    print("stepping back...")
                    print(node[1])
                    attempts = attempts + 1
                    print(points_3d_a[neighbor[0].ind])
                    print(step_back)
                    if attempts > 10:
                        new_dist = np.linalg.norm(node[1] - tmp_point)
                        for tmp_ind in range(len(node[0].neighbors)):
                            if node[0].neighbors[tmp_ind][0].ind == neighbor[0].ind:
                                node[0].neighbors[tmp_ind] = (neighbor[0], new_dist)
                        for tmp_ind in range(len((neighbor[0].neighbors))):
                            if neighbor[0].neighbors[tmp_ind][0].ind == node[0].ind:
                                neighbor[0].neighbors[tmp_ind] = (node[0], new_dist)
                        toDo[neighbor[0].ind] = (neighbor[0], tmp_point)
                    else: 
                        toDo = {}
                        done = {}
                        toDo[node[0].ind] = (node[0], step_back)
                        break

    points_3d_b = np.zeros(points_3d_a.shape)
    for ind in done:
        points_3d_b[ind] = done[ind][1]
        #points_3d_b[ind] = closest_on_ray(points_3d_a[ind], rays[ind]) 
    return points_3d_b

# Find the closest point on a ray to a given 3d point 
def closest_on_ray(point_3d, ray):
    v1 = point_3d - ray.o
    #theta = np.arccos(np.dot(ray.dir, v1)/ (hyp * np.linalg.norm(ray.dir)))
    #t = np.cos(theta) * hyp
    t = np.dot(ray.dir, v1)
    return (t * ray.dir) + ray.o

# This function will find the index of the pair of points in these
# sets that are closest to each other in euclidean distance
def find_closest_ind(points_2d_a, points_2d_b):
    return np.argmin(np.sum(np.square(points_2d_a - points_2d_b), axis=1))
    pass

# This function will find the point that lies on a given ray that has
# a specified distance away from a given point and is the closest
def find_next_on_ray(point_3d, point_3d_old, ray, dist):
    closest = closest_on_ray(point_3d, ray)
    v1 = closest - point_3d
    closest_dist = np.linalg.norm(v1)
    if np.abs(closest_dist - dist) <= 0.001:
        return None, closest
    elif closest_dist >= dist:
        #print("in function")
        #print(ray.o)
        #print(dist)
        #print(point_3d)
        v2 = point_3d - ray.o
        #norm_v2 = v2/np.linalg.norm(v2)

        # This is the case where we step backward
        #print("in function")
        #print(dist/np.sin(np.arccos(np.clip(np.dot(ray.dir, norm_v2), -1.0, 1.0))))
        #print(dist)
        #print(closest_dist)
        return (dist/closest_dist) * v2 + ray.o, closest
        #return 0.99 * norm_v2 * dist/np.sin(np.arccos(np.clip(np.dot(ray.dir, norm_v2), -1.0, 1.0))), None
        #return ((v1 / closest_dist) * dist) + point_3d
    else:
        t = np.sin(np.arccos(closest_dist/dist)) * dist
        a = closest + (ray.dir * t)
        b = closest - (ray.dir * t)
        if np.sum(np.square(a - point_3d_old)) > np.sum(np.square(b - point_3d_old)):
            return None, b
        else:
            return None, a
    pass



# This function will take in 2d points and shoot rays 
def points_to_rays(points_2d):
    camera_dist = 1080.0 #np.amax(np.flatten(points_2d))
    points = np.insert(points_2d, 2, camera_dist, axis=1)
    rays = []
    for i in range(points_2d.shape[0]):
        tmp = points[i] - np.array([960.0, 540.0, 0.0])
        direction = tmp / np.linalg.norm(tmp)
        ray = Ray(np.array([960.0 , 540.0, 0.0]), direction)
        rays.append(ray)
    return rays, points

