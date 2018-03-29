import numpy as np
# from scanner import Block

# def make_lors(blockpairs):
#     lors = []
#     for ibp in blockpairs:
#         b0 = ibp[0]
#         b1 = ibp[1]
#         m0 = b1.meshes()
#         m1 = b2.meshes()
#         lors.append(list(itertools.product(m0, m1)))
#     return lors


def partition_lors(lors: np.ndarray):
    """
    patition the input lors into three np.array
    according to the dominant direction.
    """
    dim_size1 = (lors.shape)[1]
    lors.reshape((-1, dim_size1))
    # lors = np.reshape(lors, (-1, dim_size1))
    p1 = lors[:, 0:3]
    p2 = lors[:, 3:6]

    # find the index of x_main, y_main and z_main
    diff = np.abs(p2 - p1)
    x_d = diff[:, 0]
    y_d = diff[:, 1]
    z_d = diff[:, 2]
    # x_mask = np.logical_and(x_d >= y_d,x_d >= z_d)
    # y_mask = np.logical_and(np.logical_and(x_d < y_d,y_d >= z_d), np.logical_not(x_mask))
    # z_mask = np.logical_not(np.logical_or(x_mask, y_mask))
    # xlors = lors[x_mask, :]
    # ylors = lors[y_mask, :]
    # zlors = lors[z_mask, :]
    xlors = lors[np.array([np.intersect1d(np.where(x_d >= y_d),
                                          np.where(x_d >= z_d))])].reshape((-1, dim_size1))
    ylors = lors[np.array([np.intersect1d(np.where(y_d > x_d),
                                          np.where(y_d >= z_d))])].reshape((-1, dim_size1))
    zlors = lors[np.array([np.intersect1d(np.where(z_d > x_d),
                                          np.where(z_d > y_d))])].reshape((-1, dim_size1))
    return xlors, ylors, zlors

# exchange the start point and end point if the


def swap_points(lors: np.ndarray, axis: int):
    """
    lors is an 2D array whose shape is [n, 6]
    Two points are indicated 
    in the order of (x1, y1, z1, x2, y2, z2).
    This function garantees the point2 is 
    large than the point1 in main axis(int (0, 1, 2) for (x, y, z))
    """
    dim_size1 = (lors.shape)[1]
    if axis > 2 or axis < 0:
        raise ValueError
    point1 = lors[:, 0:3]
    point2 = lors[:, 3:6]
    # print("point1")
    # print(point1)
    # print('point2')
    # print(point2)
    d = point2[:, axis] - point1[:, axis]
    # print("diff....")
    # print(np.array([np.where(d < 0)]))
    positive = lors[np.array([np.where(d >= 0)])].reshape((-1, dim_size1))
    negative = lors[np.array([np.where(d < 0)])].reshape((-1, dim_size1))
    # print("negative....")
    # print(negative)
    # print("positive")
    # print(positive)
    swaped_nega = np.hstack((np.array(negative[:, 3:6]),
                             np.array(negative[:, 0:3]),
                             np.array(negative[:, 6:])))
    return np.vstack((positive, swaped_nega))

    # negative[:,0:3], negative[:, 3:6] = negative[:,3:6], negative[:, 0:3]
    # swaped_n = np.hstack[]
    # swaped_lors = (positive, negative)
    # return swaped_lors
def preprocess(lors:np.ndarray):
    xlors, ylors, zlors = partition_lors(lors)
    xlors = swap_points(xlors, 0)
    ylors = swap_points(ylors, 1)
    zlors = swap_points(zlors, 2)
    return xlors, ylors, zlors