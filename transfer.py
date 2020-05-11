import numpy as np

# Represents a ray (in hindsight, since we always use the same origin, we could modify
# this to not contain the origin and just have a predifined camera origin, oh well
#   o : the origin as a vector of length 3
#   dir : the direction as a vector of length 3 (should be normalized)
class Ray:
    def __init__(self, o, direction):
        self.dir = direction
        self.o = o

# Represents a node in our graph (i.e., a joint that has fixed distances to other joints)
#   ind : the index in the point array
#   neighbors : a list of tuples, first item is a neighbor and second item is the distance to that neighbor
class GraphNode:
    def __init__(self, ind):
        self.neighbors = []
        self.ind = ind
    
    # Adds a neighbor, and then will add itself as a neighbor to the neighbor
    #   neighbor : the GraphNode neighbor
    #   points : 2D positions of the points
    #   recur : whether or not to make a recursive call (add yourself as a neighbor)
    def add_neighbor(self, neighbor, points, recur):
        self.neighbors.append((neighbor, np.linalg.norm(points[neighbor.ind] - points[self.ind])))
        if recur:
            neighbor.add_neighbor(self, points, False)

# Build a graph given 2d points, note that we hard-code the connectivity here
# returns : a list of GraphNode
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

    


# Finds the initial graph and 3D points for the first frame
def find_initial(points_2d):
    graph = build_graph(points_2d)
    _, points = points_to_rays(points_2d)
    return points, graph


# Finds the next set of 3d points given an initial frame with 2d and 3d points,
# the next set of 2d points, and the graph describing
# the fixed distances between the points
def find_next(points_2d_a, points_2d_b, points_3d_a, graph):
    # Not sure if this is the best option, but trying it for now, this is the one we use to assume our starting point
    closest_ind = find_closest_ind(points_2d_a, points_2d_b)
    # Find all the rays
    rays, _ = points_to_rays(points_2d_b)
    # find the closest point for the starting point we selected
    closest = closest_on_ray(points_3d_a[closest_ind], rays[closest_ind])
    # nodes we have visited already and have their neighbors visited
    done = {}
    # nodes we have visited already, but still need to check if we visited their neighbors too
    toDo = {closest_ind : (graph[closest_ind], closest)}
    # keep track of the number of times we "step_back" when the closest distance is too far
    attempts = 0
    while len(toDo) > 0:
        # Get an arbitrary node
        ind = list(toDo.keys())[0]
        node = toDo.pop(ind)
        done[ind] = node
        # Iterate through the neighbors to find their estimated 3D positions, then add them to todo
        for neighbor in node[0].neighbors:
            if not (neighbor[0].ind in done or neighbor[0].ind in toDo):
                step_back, tmp_point = find_next_on_ray(node[1], points_3d_a[neighbor[0].ind], rays[neighbor[0].ind], neighbor[1])
                # In this case, we are neither updating the source 3D position nor the distances in our graph 
                if step_back is None:
                    toDo[neighbor[0].ind] = (neighbor[0], tmp_point)
                else:
                    attempts = attempts + 1
                    if attempts > 10:
                        # If we tried updating one of our 3D positions at least 10 times, then we decide to accept
                        # the closest point and just update the distance in our graph
                        new_dist = np.linalg.norm(node[1] - tmp_point)
                        for tmp_ind in range(len(node[0].neighbors)):
                            if node[0].neighbors[tmp_ind][0].ind == neighbor[0].ind:
                                node[0].neighbors[tmp_ind] = (neighbor[0], new_dist)
                        for tmp_ind in range(len((neighbor[0].neighbors))):
                            if neighbor[0].neighbors[tmp_ind][0].ind == node[0].ind:
                                neighbor[0].neighbors[tmp_ind] = (node[0], new_dist)
                        toDo[neighbor[0].ind] = (neighbor[0], tmp_point)
                    else:
                        # If we have not attempted to update 3D positions 10 times, then we reset our visited dictionaries
                        # and then update the 3D position instead of any distances in our graph
                        toDo = {}
                        done = {}
                        toDo[node[0].ind] = (node[0], step_back)
                        break

    points_3d_b = np.zeros(points_3d_a.shape)
    for ind in done:
        points_3d_b[ind] = done[ind][1]
    return points_3d_b

# Find the closest point on a ray to a given 3d point 
def closest_on_ray(point_3d, ray):
    v1 = point_3d - ray.o
    t = np.dot(ray.dir, v1)
    return (t * ray.dir) + ray.o

# This function will find the index of the pair of points in these
# sets that are closest to each other in euclidean distance
def find_closest_ind(points_2d_a, points_2d_b):
    return np.argmin(np.sum(np.square(points_2d_a - points_2d_b), axis=1))

# This function will find the point that lies on a given ray that has
# a specified distance away from a given point and is the closest
def find_next_on_ray(point_3d, point_3d_old, ray, dist):
    closest = closest_on_ray(point_3d, ray)
    v1 = closest - point_3d
    closest_dist = np.linalg.norm(v1)
    if np.abs(closest_dist - dist) <= 0.001:
        # Case (1)
        return None, closest
    elif closest_dist >= dist:
        # Case (3)
        v2 = point_3d - ray.o
        return (dist/closest_dist) * v2 + ray.o, closest
    else:
        # Case (2)
        t = np.sin(np.arccos(closest_dist/dist)) * dist
        a = closest + (ray.dir * t)
        b = closest - (ray.dir * t)
        if np.sum(np.square(a - point_3d_old)) > np.sum(np.square(b - point_3d_old)):
            return None, b
        else:
            return None, a
    pass



# This function will take in 2d points and cast rays through them, returning the rays 
def points_to_rays(points_2d):
    camera_dist = 1080.0 
    points = np.insert(points_2d, 2, camera_dist, axis=1)
    rays = []
    for i in range(points_2d.shape[0]):
        tmp = points[i] - np.array([960.0, 540.0, 0.0])
        direction = tmp / np.linalg.norm(tmp)
        ray = Ray(np.array([960.0 , 540.0, 0.0]), direction)
        rays.append(ray)
    return rays, points

