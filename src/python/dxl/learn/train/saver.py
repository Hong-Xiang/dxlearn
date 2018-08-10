import tensorflow as tf
from dxpy.filesystem import Path
from ..graph import Graph
import arrow

_instance = None


from dxpy.configs import configurable
from ..config import config


def get_saver():
    global _instance
    if _instance is None:
        _instance = Saver()
    return _instance


class Saver(Graph):
    @configurable(config, with_name=True)
    def __init__(self, name='saver', model_dir='./save/', ckpt_name='save',
                 *, auto_save_interval=600, **config):
        super(__class__, self).__init__(name, model_dir=model_dir,
                                        ckpt_name=ckpt_name, auto_save_interval=auto_save_interval, **config)
        self._saver = None
        self.last_save = None
        self.register_task('save', self.save)
        self.register_task('load', self.load)

    def _model_path(self):
        return (Path(self.param('model_dir')) / self.param('ckpt_name')).abs

    def save(self, feeds):
        from ..scalar import global_step
        if self._saver is None:
            self._saver = tf.train.Saver()
        from dxpy.learn.session import get_default_session
        sess = get_default_session()
        step = sess.run(global_step())
        print("[SAVE] model to: {}.".format(self._model_path()))
        self._saver.save(sess, self._model_path(), global_step=step)

    def auto_save(self, feeds):
        if self.last_save is None or (arrow.now() - self.last_save).seconds > self.param('auto_save_interval'):
            self.save(feeds)
            self.last_save = arrow.now()

    def __resolve_path_load(self, feeds):
        from fs.osfs import OSFS
        import re
        from dxpy.filesystem import Path
        path_check_point = (
            Path(self.param('model_dir', feeds)) / 'checkpoint').abs
        pp = re.compile('^.*: "(.*)".*$')
        ps = re.compile('.*' + self.param('ckpt_name', feeds) + '-([0-9]+)-*')
        paths = []
        with OSFS('/') as fs:
            if not fs.exists(path_check_point):
                return fs.getsyspath(path_check_point), False
            with fs.open(path_check_point) as fin:
                for l in fin.readlines():
                    mat_path = pp.match(l)
                    if mat_path is not None:
                        path_load = mat_path[1]
                        mat_step = ps.match(path_load)
                        if mat_step is not None:
                            paths.append([path_load, int(mat_step[1])])
        step = self.param('step', feeds, default=-1)
        if step == -1:
            step = max(list(zip(*paths))[1])
        for p, s in paths:
            if s == step:
                return p, True
        return step, False

    def load(self, feeds):
        from ..scalar import global_global_step
        import sys
        if self._saver is None:
            self._saver = tf.train.Saver()
        from dxpy.learn.session import get_default_session
        sess = get_default_session()
        path_load, flag = self.__resolve_path_load(feeds)
        if flag is False:
            if isinstance(path_load, int):
                msg = "[ERROR][LOAD] Save for given step {} not found. Skip restore."
                print(msg.format(path_load), file=sys.stderr)
                return
            else:
                msg = "[ERROR][LOAD] Checkpoint file {} not found. Skip restore."
                print(msg.format(path_load), file=sys.stderr)
                return
        print("[LOAD] model from: {}.".format(path_load))
        self._saver.restore(sess, path_load)
