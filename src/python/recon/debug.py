import numpy as np
from preprocess import cut_lors


def test_cut_lors():
    lors = np.array([[-10., 0., 0., 10., 0., 0., 3],
                     [0., 0., -10., 0., 0., 10., 4],
                     [0., -4., 0., 0., 6., 0., 0],
                     [-5., -5., -5., 5., 5., 5., -8],])
    # lors = np.array([ ])
    # print(lors.shape)
    b = cut_lors(lors, limit=5)
    print("b:\n", b)


def bin_to_npy(file_name):
    root = '/home/chengaoyu/code/Python/gitRepository/dxlearn/develop-cgy/'
    bin_file = root +file_name
    npyfile = root + 'mouse_data.npy'
    lors = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 7))  
    np.save(npyfile, lors)

if __name__ == "__main__":
    test_cut_lors()
    # bin_to_npy('mouse_data.s')
# a = np.ones((2, 6))
# b = 0 - a
# c = np.square(b)
# d = np.sum(c, 0)
# print(b)
# print(c)
# print(d)
