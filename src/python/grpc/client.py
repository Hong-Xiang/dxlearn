import numpy as np

import grpc
import recon_pb2
import recon_pb2_grpc

MAX_MESSAGE_LENGTH = 30 * 1024 * 1024

CHUNK_SIZE = 256 * 256


def splitted_image_maker(image, message_maker):
  image = image.flatten()
  images = [image[i:i + CHUNK_SIZE] for i in range(0, image.size, CHUNK_SIZE)]
  for im in images:
    msg = message_maker()
    msg.image.extend(im)
    yield msg


def combine_image(request_iterator):
  image = None
  cpos = 0
  for req in request_iterator:
    if image is None:
      grid = req.grid
      center = req.center
      size = req.size
      image = np.zeros([grid[0] * grid[1] * grid[2]])
    image[cpos:cpos + len(req.image)] = req.image
    cpos += len(req.image)
  return image.reshape(grid)


import time


def recon(stub, effmap_file, lor_files, lor_range, image, grid, center, size):
  def payload_maker():
    req = recon_pb2.ReconPayload()
    req.efficiency_map_file = effmap_file
    req.lor_files.extend(lor_files)
    req.lor_range.extend(lor_range)
    req.grid.extend(grid)
    req.center.extend(center)
    req.size.extend(size)
    return req

  reqs = splitted_image_maker(image, payload_maker)
  res = [r for r in stub.ReconStep(reqs)]
  image = combine_image(res)
  return image


import threading
from multiprocessing import Pool


def main():
  workers = [
      # '192.168.1.118:50050',
      # '192.168.1.118:50051',
      '192.168.1.111:50050',
      '192.168.1.111:50051',
  ]
  NB_WORKERS = len(workers)
  channels = [grpc.insecure_channel(w) for w in workers]
  stubs = [recon_pb2_grpc.ReconstructionStub(c) for c in channels]
  root = '/hqlf/hongxwing/RPCRecon/debug/'
  effmap_file = root + 'map.npy'
  lor_files = [root + '{}lors.npy'.format(a) for a in ['x', 'y', 'z']]
  lor_step = int(1e6)
  lor_range = [0, lor_step]
  # grid = [90, 110, 130]
  grid = [150, 150, 150]
  center = [0., 0., 0.]
  # size = [90., 110., 130.]
  size = [150., 150., 150.]
  image = np.ones(grid)
  # root = './debug/'
  images = [None for i in range(NB_WORKERS)]

  def recon_thrd(iworker):
    s = stubs[iworker]
    lor_range = [lor_step * iworker, lor_step * (iworker + 1)]
    result = recon(s, effmap_file, lor_files, lor_range, image, grid, center,
                   size)
    images[iworker] = images

  st = time.time()

  for i in range(20):
    image = recon(stubs[0], effmap_file, lor_files, lor_range, image, grid,
                  center, size)
    # with Pool(NB_WORKERS) as p:
    # images = p.map(recon_thrd, range(NB_WORKERS))
    # threads = [
    #     threading.Thread(target=recon_thrd, args=(iw, ))
    #     for iw in range(NB_WORKERS)
    # ]
    # # return result
    # for t in threads:
    #   t.start()
    # for t in threads:
    #   t.join()
    # image = np.sum(images, axis=0)
    # print(result)
    np.save('./debug/rpc_result_{}.npy'.format(i), image)
    et = time.time()
    print('RUN: {} s, PER IT: {} s.'.format(et - st, (et - st) / (i + 1)))
  print('DONE!')


if __name__ == "__main__":
  main()