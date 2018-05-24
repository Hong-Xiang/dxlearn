from ..graph import Graph
from ..model.image import MultiDownSampler
from dxpy.configs import configurable
from ..config import config


class SuperResolutionDataset(Graph):
    """
    SuperResolutionDataset is a special kind of dataset which
    """
    @configurable(config, with_name=True)
    def __init__(self, name, origial_dataset_maker, input_key=None, nb_down_sample=None, with_shape_info=None, origin_shape=None, **config):
        super().__init__(name, input_key=input_key,
                         nb_down_sample=nb_down_sample,
                         with_shape_info=with_shape_info,
                         origin_shape=origin_shape,
                         **config)
        self._dataset_maker = origial_dataset_maker
        self.__construct()

    @classmethod
    def _default_config(cls):
        from dxpy.collections.dicts import combine_dicts
        return combine_dicts({'with_shape_info': False}, super()._default_config())

    def __down_sample_keys(self, start):
        return [('image{}x'.format(2**i), 2**i) for i in range(start, self.param('nb_down_sample') + 1)]

    def __construct(self):
        self.register_node('dataset',
                           self._dataset_maker())
        down_sample_ratios = {k: [v] * 2 for k,
                              v in self.__down_sample_keys(1)}
        origin_key = self.__down_sample_keys(0)[0][0]
        self.register_node('down_sampler',
                           MultiDownSampler(self.nodes['dataset'][self.param('input_key')],
                                            down_sample_ratios=down_sample_ratios,
                                            keep_original=True, original_key=origin_key,
                                            register_output_with_prefix=False))
        for k, _ in self.__down_sample_keys(0):
            self.register_node(k, self.nodes['down_sampler'][k])


def random_crop_multiresolution(images, target_shape_high_resolution, nb_down_sample=None):
    """
    Args:
        images: dict-like object. Containing 'image{0,1,2,3,...}x'.
    """
    from dxpy.learn.utils.tensor import shape_as_list
    import tensorflow as tf
    if nb_down_sample is None:
        nb_down_sample = len(images) - 1

    with tf.name_scope('random_crop_multi_resolution'):
        shapes = [shape_as_list(images['image{}x'.format(i)])
                  for i in range(nb_down_sample + 1)]
        down_sample_ratios = [sh // sl for sh, sl in zip(shapes[0], shapes[1])]
        for i in range(len(shapes) - 1):
            for sh, sl, dr in zip(shapes[i], shapes[i + 1], down_sample_ratios):
                if sl * dr != sh:
                    raise ValueError(
                        "Invalid inputs shapes: {}".format(shapes))
        for sh, st in zip(shapes[0], target_shape_high_resolution):
            if sh < st:
                raise ValueError("Can not perfrom random crop on inputs with shape {} with target shape {}.".format(
                    shapes, target_shape_high_resolution))
        target_shapes = []
        for i in range(nb_down_sample + 1):
            target_shapes.append([s // (dr**i)
                                  for s, dr in zip(target_shape_high_resolution, down_sample_ratios)])
        max_offsets = [sl - tl for sl,
                       tl in zip(shapes[-1], target_shapes[-1])]
        offset = []
        for mo in max_offsets:
            if mo == 0:
                offset.append(0)
            else:
                offset.append(tf.random_uniform([], 0, mo, dtype=tf.int64))
        offsets = [offset]
        for i in range(nb_down_sample):
            offset_new = [o * dr for o,
                          dr in zip(offsets[-1], down_sample_ratios)]
            offsets.append(offset_new)
        offsets = list(offsets[::-1])
        results = dict()
        for i in range(nb_down_sample + 1):
            k = 'image{}x'.format(i)
            results[k] = tf.slice(images[k], offsets[i], target_shapes[i])
        return results
