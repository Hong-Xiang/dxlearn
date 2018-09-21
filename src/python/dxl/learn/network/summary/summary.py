import tensorflow as tf
# from dxl.learn.core import Graph, Tensor, ThisSession
from dxl.learn.core import Tensor, ThisSession
from dxl.learn.graph import Graph
# from dxl.learn.tensor._global_step import GlobalStep


class SummaryItem(Tensor):
    summary_func = None

    def __init__(self, name, data):
        """
        `info`: GraphInfo object.
        """
        self.name = name
        self.data = data

    def make(self):
        pass


class ScalarSummary(SummaryItem):
    def make(self):
        return tf.summary.scalar(self.name, self.data)


class ImageSummary(SummaryItem):
    def __init__(self, name, data, nb_max_outs=3):
        super().__init__(name, data)
        self.nb_max_outs = nb_max_outs

    def make(self):
        return tf.summary.image(self.name, self.data, )


class SummaryWriter(Graph):
    """
    SummaryWriter is a warp over tf.SummaryWriter, it is a speical kind of Graph
    which only accepts SummaryTensors.
    SummaryTensors is a special kind of tensor data :: tf.Tensor and
    summary_name: str.

    If tensors is `None`, SummaryWriter just write graph definition to file.
    This is useful when debugging network architecture, in this case,
    one simply use SummaryWritter(path='/tmp/debug/', depsession=sess)
    """

    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            PATH = 'path'
            NB_INTERVAL = 'nb_iterval'
            NB_MAX_IMAGE = 'nb_max_image'
            PREFIX = 'prefix'

    def __init__(self,
                 info='summary_writter',
                 path=None,
                 nb_iterval=None,
                 *,
                 nb_max_image=None,
                 prefix=None,):
        super().__init__(info, config={
            self.KEYS.CONFIG.PATH: path,
            self.KEYS.CONFIG.NB_INTERVAL: nb_iterval,
            self.KEYS.CONFIG.NB_MAX_IMAGE: nb_max_image,
            self.KEYS.CONFIG.PREFIX: prefix
        })
        self.file_writer = tf.summary.FileWriter(path)
        self._next_summary_step = 0

    def close(self):
        self.file_writer.close()

    def add_item(self, t):
        if t.name in self.tensors:
            raise ValueError("{} already in tensors.".format(t.name))
        self.tensors[t.name] = t

    def add_loss(self, loss):
        self.add_item(ScalarSummary('loss', loss))

    def add_graph(self, g=None):
        if g is None:
            g = tf.get_default_graph()
        self.file_writer.add_graph(g)

    def kernel(self, inputs=None):
        summary_ops = []

        for t, v in self.tensors.items():
            summary_ops.append(v.make())
        self.summary_op = tf.summary.merge(summary_ops)

    def run(self, feeds=None):
        """
        Run tensors and dump to summary file.
        """
        self.make()
        result = ThisSession.run(self.summary_op, feeds)
        self.file_writer.add_summary(result, ThisSession.run(GlobalStep()))

    def auto_run(self, feeds=None):
        if ThisSession.run(GlobalStep()) >= self._next_summary_step:
            self.run(feeds)
            self._next_summary_step += self.config(
                self.KEYS.CONFIG.NB_INTERVAL)

    @property
    def summary_step(self):
        return self.config(self.KEYS.CONFIG.NB_INTERVAL)
