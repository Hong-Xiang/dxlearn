import numpy as np
from preprocess import cut_lors


def test_cut_lors():
    lors = np.array([[-10., 0., 0., 10., 0., 0., 3],
                     [0., 0., -10., 0., 0., 10., 4],
                     [0., -4., 0., 0., 6., 0., 0],
                     [-5., -5., -5., 5., 5., 5., 0],])
    # lors = np.array([ ])
    # print(lors.shape)
    b = cut_lors(lors, limit=5)
    print("b:\n", b)


if __name__ == "__main__":
    test_cut_lors()
# a = np.ones((2, 6))
# b = 0 - a
# c = np.square(b)
# d = np.sum(c, 0)
# print(b)
# print(c)
# print(d)
