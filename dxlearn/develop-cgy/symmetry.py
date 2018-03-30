import numpy as np
import itertools
"""
the block index 0 start from the x axis
"""


def make_table(block_vector):
    table = np.asarray(list(itertools.product(block_vector, block_vector))).reshape(
        num_blocks, num_blocks, 2)
    return table



def rotate_quater(block_vector):
    num_blocks = len(block_vector)
    # print(num_blocks)
    num_quater = np.int(num_blocks/4)
    # print(num_quater)
    return np.hstack((block_vector[num_quater:num_blocks],
                      block_vector[0:num_quater]))


def flip_x(block_vector):
    num_blocks = len(block_vector)
    num_quater = np.int(num_blocks/4)
    q1 = block_vector[0: num_quater]
    q2 = block_vector[num_quater: 2*num_quater]
    q3 = block_vector[2*num_quater: 3*num_quater]
    q4 = block_vector[3*num_quater: num_blocks]

    rq1 = np.flip(q2, 0)
    rq2 = np.flip(q1, 0)
    rq3 = np.flip(q4, 0)
    rq4 = np.flip(q3, 0)
    return np.hstack((rq1, rq2, rq3, rq4))

def flip_y(block_vector):
    num_blocks = len(block_vector)
    num_quater = np.int(num_blocks/4)
    q1 = block_vector[0: num_quater]
    q2 = block_vector[num_quater: 2*num_quater]
    q3 = block_vector[2*num_quater: 3*num_quater]
    q4 = block_vector[3*num_quater: num_blocks]

    rq1 = np.flip(q4, 0)
    rq2 = np.flip(q3, 0)
    rq3 = np.flip(q2, 0)
    rq4 = np.flip(q1, 0)
    return np.hstack((rq1, rq2, rq3, rq4))

# def flip_xy(block_vector):
#     """
#     flip along x = y
#     """
#     pass

# def flip_xry()
#     """
#     flip along x = -y
#     """


if __name__ == '__main__':
    num_blocks = 20

    block_vector = np.arange(num_blocks)
    block_phi =  block_vector*np.pi/(num_blocks)
    # table = np.asarray(list(itertools.product(block_list, block_list))).reshape(
    #     num_blocks, num_blocks, 2)
    # print(table)
    print(block_phi)
    # print(rotate_quater(block_vector))
    # print(flip_x(block_vector))
    # print(flip_y(block_vector))
    # print(np.flip(np.arange(5), 0))
