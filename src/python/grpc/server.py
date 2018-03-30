import grpc
import h5py
import time
import recon_pb2
import recon_pb2_grpc
from concurrent import futures
import numpy as np
<<<<<<< HEAD


def reconstruct_step(efficiency_map_file, lor_file, lor_start, lor_end, image):
  # with h5py.File(efficiency_map_file, 'r') as fin:
  #   effmap = np.array(fin['data'])
  # with h5py.File(lor_file, 'r') as fin:
  #   lors = np.array(fin['data'][lor_start:lor_end, :])
  effmap = np.load(efficiency_map_file)
  lors = np.load(lor_file)[lor_start:lor_end, :]
  return image / effmap + np.sum(lors) / image.size
=======
from dxl.learn.high_level.reconstep import recon_step


def reconstruct_step(efficiency_map_file, lor_files, lor_range, image, center, size):
  print(efficiency_map_file)
  print(lor_files)
  print(lor_range)
  print(center)
  print(size)
  if efficiency_map_file.endswith('.h5'):
    with h5py.File(efficiency_map_file, 'r') as fin:
      effmap = np.array(fin['data'])
  elif efficiency_map_file.endswith('.npy'):
    effmap = np.load(efficiency_map_file)
  lors = []
  for lor_file in lor_files:
    if lor_file.endswith('.h5'):
      with h5py.File(lor_file, 'r') as fin:
        lors.append(np.array(fin['data'][lor_start:lor_end, :]))
    elif lor_file.endswith('.npy'):
      lors.append(np.load(lor_file)[lor_start:lor_end, :])
  image = recon_step(efficiency_map, lors, image, center, size)
  return image 
>>>>>>> 69901e0fe34721a58822a0dddaeee48b5309f8e7


class ReconstructionService(recon_pb2_grpc.ReconstructionServicer):
  def ReconStep(self, request, context):
    result = recon_pb2.Image()
<<<<<<< HEAD
    image_shape = list(request.image_shape)
    result_image = reconstruct_step(
        request.efficiency_map_file, request.lor_file, request.lor_start,
        request.lor_end,
        np.array(request.image).reshape(image_shape))
    result.values.extend(result_image.flatten())
=======
    result_image = reconstruct_step(
        request.efficiency_map_file, request.lor_files, request.lor_range,
        np.array(request.image).reshape(request.grid),
        request.center, request.size)
    result.image.extend(result_image.flatten())
    result.grid.extend(image.shape)
    result.center.extend(request.center)
    result.size.extend(request.size)
>>>>>>> 69901e0fe34721a58822a0dddaeee48b5309f8e7
    return result


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
