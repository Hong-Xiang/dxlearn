import numpy as np
import h5py
import tensorflow as tf
from ...core import ThisHost


class ImageInfo:
  def __init__(self, grid, center, size):
    self.grid = grid
    self.center = center
    self.size = size


class DataInfo:
  def __init__(self, map_file, lor_files, lor_ranges=None, lor_step=None):
    self._map_file = map_file
    self._lor_files = lor_files
    self._lor_ranges = lor_ranges
    self._lor_step = lor_step

  def map_file(self):
    if isinstance(self._map_file, str):
      return self._map_file
    else:
      return self._map_file[ThisHost.host().task_index]

  def lor_file(self, axis):
    if isinstance(self._lor_files[axis], str):
      return self._lor_files[axis]
    else:
      return self._lor_files[axis][ThisHost.host().task_index]

  def lor_range(self):
    tid = ThisHost.host().task_index
    if self._lor_ranges is not None:
      return self._lor_ranges[tid]
    elif self._lor_step is not None:
      return [tid * self._lor_step, (tid + 1) * self._lor_step]
    else:
      return None


def load_data(file_name, lor_range=None):
  if file_name.endswith('.npy'):
    data = np.load(file_name)
    if lor_range is not None:
      data = data[lor_range[0]:lor_range[1], :]
  elif file_name.endswith('.h5'):
    with h5py.File(file_name, 'r') as fin:
      if lor_range is not None:
        data = np.array(fin['data'][lor_range[0]:lor_range[1], :])
      else:
        data = np.array(fin['data'])
  return data


def ensure_float32(x):
  if isinstance(x, np.ndarray) and x.dtype == np.float64:
    return np.array(x, dtype=np.float32)
  return x


def constant_tensor(x, name, ginfo):
  return TensorNumpyNDArray(_ensure_float32(x), None, ginfo.update(name=name))


def variable_tensor(x, name, ginfo):
  x_tensor = TensorVariable(
      VariableInfo(None, _ensure_float32(x), tf.float32),
      ginfo.update(name=name))
  x_init = x_tensor.assign(_constant_tensor(x, name + '_initial_value', ginfo))
  return x_tensor, x_init


def print_tensor(t, name=None):
  print("[DEBUG] name: {}, tensor: {}, value:\n{}".format(
      name, t.data, t.run()))


def print_info(*msg):
  print('INFO', *msg)


sample_cluster_config = {
    "master": ["localhost:2221"],
    "worker": ["localhost:2333", "localhost:2334"]
}


def cluster_configs_generator(master_ip, workers_ips, nb_process_per_worker):
  workers = []
  for worker in WORKERS_IO_SUFFIX:
    for p in range(2333, 2333 + NB_PROCESS_PER_WORKER):
      workers.append("192.168.1.{}:{}".format(worker, p))
  return {
      "master": ["192.168.1.{}:2221".format(MASTER_IP_SUFFIX)],
      "worker": workers
  }


def load_cluster_configs(config=None):
  if config is None:
    return sample_cluster_config
  elif isinstance(config, str):
    with open(config, 'r') as fin:
      return json.load(fin)
  else:
    return config


sample_reconstruction_config = {
    'grid': [150, 150, 150],
    'center': [0., 0., 0.],
    'size': [150., 150., 150.],
    'map_file': './debug/map.npy',
    'x_lor_files': './debug/xlors.npy',
    'y_lor_files': './deubg/ylors.npy',
    'z_lor_files': './deubg/zlors.npy',
    'lor_ranges': None,
    'lor_steps': None,
}


def load_reconstruction_configs(config=None):
  if config is None:
    c = sample_cluster_config
  elif isinstance(config, str):
    with open(config, 'r') as fin:
      c = json.load(fin)
  else:
    c = config
  image_info = ImageInfo(c['grid'], c['center'], c['size'])
  data_info = DataInfo(c['map_file'],
                       {a: c['{}_lor_files']
                        for a in ['x', 'y', 'z']}, lor_ranges, lor_steps)
  return image_info, data_info