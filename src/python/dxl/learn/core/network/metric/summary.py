import tensorflow as tf
from .graph import Graph
from .tensor import Tensor


class SummaryTensor(Tensor):
    summary_func = None
    def __init__(self, summary_name, data, info):
        """
        `info`: GraphInfo object.
        """
        self.summary_name = summary_name
        pass

class ScalarSummary(SummaryTensor):
    summary_func = tf.summary.scalar 

class ImageSummary(SummaryTensor):
    summary_func = tf.summary.image

class SummaryWriter(Graph):
    """
    SummaryWriter is a warp over tf.SummaryWriter, it is a speical kind of Graph
    which only accepts SummaryTensors.
    SummaryTensors is a special kind of tensor data :: tf.Tensor and
    summary_name: str.

    If tensors is `None`, SummaryWriter just write graph definition to file.
    This is useful when debugging network architecture, in this case,
    one simply use SummaryWritter(path='/tmp/debug/', session=sess)
    """

    def __init__(self, name='summary', tensors=None,
                session=None,
                 *,
                 nb_iterval=None,
                 method_method=None,
                 path=None,
                 nb_max_image=None,
                 with_prefix=None,):
        pass
