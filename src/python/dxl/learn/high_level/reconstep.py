import numpy as np
from dxl.learn.core import TensorNumpyNDArray, GraphInfo, ThisSession, make_session
from dxl.learn.model.tor_reconstep import ReconStep
from typing import List


def recon_step(efficiency_map: np.ndarray, lors: List[np.ndarray],
               image: np.ndarray, center: List[float], size: List[float]):
  """
    A single reconstruction step.
    """
  efficiency_map = efficiency_map.astype(np.float32)
  image = image.astype(np.float32)
  lors = [l.astype(np.float32) for l in lors]
  ginfo = GraphInfo(None, 'reconstep')
  recon = ReconStep(
      'recon',
      TensorNumpyNDArray(image, None, ginfo.update(name='image')),
      TensorNumpyNDArray(efficiency_map, None, ginfo.update(name='emap')),
      image.shape,
      center,
      size,
      TensorNumpyNDArray(lors[0], None, ginfo.update(name='xlor')),
      TensorNumpyNDArray(lors[1], None, ginfo.update(name='ylor')),
      TensorNumpyNDArray(lors[2], None, ginfo.update(name='zlor')),
      ginfo.update(name=None),
  )()
  make_session()
  result = recon.run()
  ThisSession.reset()
  return result


if __name__ == "__main__":
  root = '/home/chengaoyu/code/Python/gitRepository/dxlearn/develop-cgy/'
  emap = np.load(root+'map.npy').T
  lors = np.load(root+'lors.npy')
  lors = lors[:int(1e6), :]
  grid = [150, 150, 150]
  center = [0., 0., 0.]
  size = [150., 150., 150.]

  img = np.ones(grid)
  from dxl.learn.preprocess import preprocess
  xlors, ylors, zlors = preprocess(lors)
  img_new = recon_step(emap, [xlors, ylors, zlors], img, center, size)
  np.save(root+'img.npy', img_new)
