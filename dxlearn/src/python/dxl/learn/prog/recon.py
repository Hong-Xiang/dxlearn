import numpy as np
import tensorflow as tf
from dxl.learn.core import TensorNumpyNDArray, TensorVariable, VariableInfo, GraphInfo, ThisSession, Session
from dxl.learn.model.recon import ReconStep, ProjectionSplitter, EfficiencyMap
from dxl.learn.model.on_collections import Summation

NB_WORKERS = 2


def init():
    root = '/mnt/gluster/hongxwing/recon_test/'
    phantom = np.load(root + 'phantom_64.0.npy')
    x = phantom.reshape([phantom.size, 1]).astype(np.float32)
    system_matrix = np.load(root + 'system_matrix_64.npy').astype(np.float32)
    y = np.matmul(system_matrix, x).astype(np.float32)
    effmap = np.matmul(system_matrix.T, np.ones(y.shape)).astype(np.float32)
    x_ = np.ones(x.shape, np.float32)

    x_var_info = VariableInfo(None, x_.shape, tf.float32)
    x_t = TensorVariable(x_var_info, GraphInfo('x_', 'var'))
    x_init = x_t.assign(TensorNumpyNDArray(
        x_, None, x_t.graph_info.update(name='x_init')))
    e_t = TensorNumpyNDArray(
        effmap, None, x_t.graph_info.update(name='effmap'))
    y_t = TensorNumpyNDArray(y, None, x_t.graph_info.update(name='y'))
    sm_t = TensorNumpyNDArray(
        system_matrix, None, x_t.graph_info.update(name='system_matrix'))
    return x_t, y_t, sm_t, e_t, x_init


def split(y_t, sm_t):
    spt = ProjectionSplitter('splitter', NB_WORKERS,
                             y_t.graph_info.update(name=None))
    y_ts = spt(y_t)
    sm_ts = spt(sm_t)
    y_ts = [y_ts['slice_{}'.format(i)] for i in range(NB_WORKERS)]
    sm_ts = [sm_ts['slice_{}'.format(i)] for i in range(NB_WORKERS)]
    return y_ts, sm_ts


def recon(x_t, y_t, sm_t, e_t, work_suffix):
    # e_t = EfficiencyMap('effmap_{}'.format(work_suffix),
                        # x_t.graph_info.update(name=None))()
    x_n = ReconStep('recon_step_{}'.format(work_suffix),
                    x_t, y_t, sm_t, e_t,
                    x_t.graph_info.update(name=None))()
    return x_n


def main():
    x_t, y_t, sm_t, e_t, x_init = init()
    y_ts, sm_ts = split(y_t, sm_t)
    ThisSession.set_session(Session())
    x_init.run()
    res = ThisSession.run(x_t.data)
    x_ns = []
    for i in range(NB_WORKERS):
        x_ns.append(recon(x_t, y_ts[i], sm_ts[i],  e_t, i))
    sm = Summation('summation', x_t.graph_info.update(name=None))
    x_n = sm(x_ns)
    x_update = x_t.assign(x_n)
    for i in range(100):
        x_update.run()
    res = x_t.run()
    print(res)
    np.save('recon.npy', res)


if __name__ == "__main__":
    main()
