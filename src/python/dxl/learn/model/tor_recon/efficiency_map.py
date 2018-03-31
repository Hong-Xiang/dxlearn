from ...core import Model, Tensor, tf_tensor
from typing import Dict
from .utils import ImageInfo, map_dict, ALL_AXIS_VALUES


def efficiency_map(lors: Dict[Tensor], image_info: ImageInfo):
  imgz = tf.zeros(grid[::-1], tf.float32)
  imgx = tf.transpose(imgz, perm=[2, 0, 1])
  imgy = tf.transpose(imgz, perm=[1, 0, 2])
  imgs = {
    'x': imgx,
    'y': imgy,
    'z': imgz,
  }
  lors = map_dict(tf_tensor, lors)
  lors = map_dict(tf.transpose, lors)

  projs = map_dict(lambda lor: tf.ones(lor.shape.as_list()[1], 1))

  model = 'tor'
  kernel_width = np.sqrt(3.4 * 3.4 / np.pi)

  grid = image_info.grid
  center = image_info.center
  size = image_info.size
  grids = {
    'x' : tf.constant(
      np.array([grid[1], grid[2], grid[0]], dtype=np.int32), name='gridx'),
    'y':  tf.constant(
      np.array([grid[0], grid[2], grid[1]], np.int32), name='gridy'),
    'z': tf.constant(
      np.array(grid, np.int32), name='gridz'),
    )
  centers = {
    'x':tf.constant(
      np.array([center[1], center[2], center[0]], np.float32), name='centerx'),
    'y':tf.constant(
      np.array([center[0], center[2], center[1]], np.float32), name='centery'),
    'z':center,
  } 
  sizez = {
    'x': tf.constant(
      np.array([size[1], size[2], size[0]], np.float32), name='sizex'),
    'y':tf.constant(
      np.array([size[0], size[2], size[1]], np.float32), name='sizey'),
    'z':size,
  }
  bps = {}
  bps = {backprojection(image=images[a]
  grid=grids[a],
  lors=lors[a],
  center=centers[a],
  size=sizes[a],
  line_integral=projs[a],
  kernel_width=kernel_width,
  model=model)
  for a in ALL_AXIS_VALUES
  }

  bps['x'] = tf.transpose(bps['x'], perm=[1, 2, 0])
  bps['y'] = tf.transpose(bps['y'], perm=[1, 0, 2])

  result = bpxt + bpyt + bpz
  # result = bpxt + bpyt
  result = tf.transpose(result)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    result = sess.run(result)
  tf.reset_default_graph()
  return result


class EfficiencyMap(Model):
  pass
