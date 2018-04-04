import numpy as np
from preprocess import cut_lors
root = '/home/chengaoyu/code/Python/gitRepository/dxlearn/develop-cgy/'


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
    # root = '/home/chengaoyu/code/Python/gitRepository/dxlearn/develop-cgy/'
    bin_file = root +file_name
    npyfile = root + 'mouse_data.npy'
    lors = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 7))  
    np.save(npyfile, lors)

if __name__ == "__main__":
    # test_cut_lors()
    effmap = np.load(root+'effmaps/siddon_1_4.npy')
    print(np.min(effmap))
    effmap = 1/effmap
    print(np.max(effmap))
    # effmap[np.array([np.where(effmap == np.nan)])]
    # print()
    # print(map)
    # bin_to_npy('mouse_data.s')

