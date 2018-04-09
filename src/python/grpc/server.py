import grpc
import h5py
import time
import recon_pb2
import recon_pb2_grpc
from concurrent import futures
import numpy as np
# from dxl.learn.high_level.reconstep import recon_step
from dxl.learn.model.tor_recon import recon_step

CHUNK_SIZE = 256 * 256


def reconstruct_step(efficiency_map_file, lor_files, lor_range, image, center,
                     size):
  print(efficiency_map_file)
  print(lor_files)
  print(lor_range)
  print(center)
  print(size)
  print(image.shape)
  if efficiency_map_file.endswith('.h5'):
    with h5py.File(efficiency_map_file, 'r') as fin:
      effmap = np.array(fin['data'])
  elif efficiency_map_file.endswith('.npy'):
    effmap = np.load(efficiency_map_file)
  effmap = effmap.T
  # effmap = effmap[::-1, ::-1, ::-1]
  lors = []
  for lor_file in lor_files:
    if lor_file.endswith('.h5'):
      with h5py.File(lor_file, 'r') as fin:
        lors.append(np.array(fin['data'][lor_range[0]:lor_range[1], :]))
    elif lor_file.endswith('.npy'):
      lors.append(np.load(lor_file)[lor_range[0]:lor_range[1], :])
  image = recon_step(effmap, lors, image, center, size)
  return image


def splitted_image_maker(image, message_maker):
  image = image.flatten()
  images = [image[i:i + CHUNK_SIZE] for i in range(0, image.size, CHUNK_SIZE)]
  messages = []
  for im in images:
    msg = message_maker()
    msg.image.extend(im)
    messages.append(msg)
  return messages


def combine_image(request_iterator):
  image = None
  cpos = 0
  for req in request_iterator:
    if image is None:
      grid = req.grid
      center = req.center
      size = req.size
      image = np.zeros([grid[0] * grid[1] * grid[2]])
      efficiency_map = req.efficiency_map_file
      lor_files = req.lor_files
      lor_range = req.lor_range
    image[cpos:cpos + len(req.image)] = req.image
    cpos += len(req.image)
  return efficiency_map, lor_files, lor_range, image.reshape(
      grid), center, size


class ReconstructionService(recon_pb2_grpc.ReconstructionServicer):
  def ReconStep(self, request_iterator, context):
    combined_req = combine_image(request_iterator)
    result_image = reconstruct_step(*combined_req)
    image_splitted = result_image.flatten()

    def make_res():
      result = recon_pb2.Image()
      result.grid.extend(result_image.shape)
      result.center.extend(combined_req[-2])
      result.size.extend(combined_req[-1])
      return result

    messages = splitted_image_maker(result_image, make_res)
    for m in messages:
      yield m


def serve():
  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=4), maximum_concurrent_rpcs=1)

  recon_pb2_grpc.add_ReconstructionServicer_to_server(ReconstructionService(),
                                                      server)

  server.add_insecure_port('[::]:50050')
  server.start()
  try:
    while True:
      _ONE_DAY = 60 * 60 * 24
      time.sleep(_ONE_DAY)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == "__main__":
  serve()
