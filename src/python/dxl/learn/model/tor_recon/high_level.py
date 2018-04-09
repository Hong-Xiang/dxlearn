import numpy as np
from ...core import TensorNumpyNDArray, GraphInfo, ThisSession, make_session
from .recon_step import ReconStep
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


def test():
  emap = np.load('./debug/map.npy').T
  lors = np.load('./debug/lors.npy')
  lors = lors[:int(1e6), :]
  grid = [150, 150, 150]
  center = [0., 0., 0.]
  size = [150., 150., 150.]

  img = np.ones(grid)
  from .utils import seperate_lors
  from tqdm import tqdm
  xlors, ylors, zlors = seperate_lors(lors)
  for i in tqdm(range(10)):
    img = recon_step(emap, [xlors, ylors, zlors], img, center, size)
    np.save('./debug/high_level_{}.npy'.format(i), img)
