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


def recon_multi(stubs, effmap_file, lor_files, lor_range):
  nb_workers = len(stubs)
  nb_lor_per_worker = (lor_range[1] - lor_range[0]) // nb_workers


def main():
  channel = grpc.insecure_channel('192.168.1.118:50050', )
  stub = recon_pb2_grpc.ReconstructionStub(channel)
  # root = './debug/'
  root = '/hqlf/hongxwing/RPCRecon/debug/'
  effmap_file = root + 'map.npy'
  lor_files = [root + '{}lors.npy'.format(a) for a in ['x', 'y', 'z']]
  lor_range = [0, int(1e5)]
  # grid = [90, 110, 130]
  grid = [150, 150, 150]
  center = [0., 0., 0.]
  # size = [90., 110., 130.]
  size = [150., 150., 150.]
  image = np.ones(grid)
  st = time.time()
  for i in range(20):
    image = recon(stub, effmap_file, lor_files, lor_range, image, grid, center,
                  size)
    # print(result)
    np.save('./debug/rpc_result_{}.npy'.format(i), image)
    et = time.time()
    print('RUN: {} s, PER IT: {} s.'.format(et - st, (et - st) / (i + 1)))
  print('DONE!')


if __name__ == "__main__":
  main()