import tensorflow as tf
from pathlib import Path
from dxl.learn.core import Graph
import arrow

_instance = None


def get_saver():
    global _instance
    if _instance is None:
        _instance = Saver()
    return _instance


class Saver(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            MODEL_DIR = 'model_dir'
            CKPT_NAME = 'ckpt_name'
            SAVE_INTERVAL = 'save_interval'
            LOAD_STEP = 'load_step'

    def __init__(self,
                 info,
                 model_dir='./model/',
                 ckpt_name='model',
                 variables=None,
                 load_step=None,
                 *,
                 save_interval=600):
        super().__init__(
            info,
            tensors=variables,
            config={self.KEYS.CONFIG.MODEL_DIR: model_dir,
                    self.KEYS.CONFIG.CKPT_NAME: ckpt_name,
                    self.KEYS.CONFIG.SAVE_INTERVAL: save_interval
                    self.KEYS.CONFIG.LOAD_STEP: load_step})
        self.last_save = None
        self.save = None

    def _model_path(self):
        return (Path(self.param('model_dir')) / self.param('ckpt_name')).abs

    def kernel(self):
        self.saver = tf.train.Saver()

    def save(self):
        from dxl.learn.tensor.global_step import GlobalStep
        from dxpy.learn.session import ThisSession
        step = ThisSession.run(GlobalStep())
        print("[SAVE] model to: {}.".format(self._model_path()))
        self.saver.save(ThisSession.session(),
                        self._model_path(), global_step=step)

    def auto_save(self):
        if self.last_save is None or (arrow.now(
        ) - self.last_save).seconds > self.config('auto_save_interval'):
            self.save()
            self.last_save = arrow.now()

    def __resolve_path_load(self):
        from fs.osfs import OSFS
        import re
        from dxpy.filesystem import Path
        path_check_point = (
            Path(self.config('model_dir')) / 'checkpoint').abs
        pp = re.compile('^.*: "(.*)".*$')
        ps = re.compile('.*' + self.config('ckpt_name') + '-([0-9]+)-*')
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
        step = self.config('step') or -1
        if step == -1:
            step = max(list(zip(*paths))[1])
        for p, s in paths:
            if s == step:
                return p, True
        return step, False

    def load(self):
        from dxl.learn.tensor.global_step import GlobalStep
        from dxl.learn.core import ThisSession
        import sys
        path_load, flag = self.__resolve_path_load()
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
        self.saver.restore(ThisSession.session(), path_load)
