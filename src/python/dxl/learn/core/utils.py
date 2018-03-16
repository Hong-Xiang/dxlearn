def map_data(list_of_tensors):
    import tensorflow as tf
    from .tensor import Tensor
    result = []
    if isinstance(list_of_tensors, (tf.Tensor, Tensor, tf.Operation)):
        list_of_tensors = [list_of_tensors]
    for t in list_of_tensors:
        if isinstance(t, Tensor):
            result.append(t.data)
        elif isinstance(t, (tf.Tensor, tf.Operation)):
            result.append(t)
        else:
            raise TypeError(
                "Unknown task tpye {}, should be Tensor or tf.Tensor.".format(type(t)))
    return result
