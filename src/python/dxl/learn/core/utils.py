import logging

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('dxl.learn')


def map_data(list_of_tensors):
  """
    Get tensorflow.Tensor for list of Tensor-like of objects.
    """
  import tensorflow as tf
  from .tensor import Tensor
  result = []
  if isinstance(list_of_tensors, (tf.Tensor, Tensor, tf.Operation)):
    # raise TypeError("map_data requires list of Tensor-like objects, maybe add a []?")
    list_of_tensors = [list_of_tensors]
  for t in list_of_tensors:
    if isinstance(t, Tensor):
      result.append(t.data)
    elif isinstance(t, (tf.Tensor, tf.Operation)):
      result.append(t)
    else:
      raise TypeError(
          "Unknown task tpye {}, should be Tensor or tf.Tensor.".format(
              type(t)))
  return result

