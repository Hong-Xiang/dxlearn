import grpc
import h5py
import time
import recon_pb2
import recon_pb2_grpc
from concurrent import futures
import numpy as np


def reconstruct_step(efficiency_map_file, lor_file, lor_start, lor_end, image):
  # with h5py.File(efficiency_map_file, 'r') as fin:
  #   effmap = np.array(fin['data'])
  # with h5py.File(lor_file, 'r') as fin:
  #   lors = np.array(fin['data'][lor_start:lor_end, :])
  effmap = np.load(efficiency_map_file)
  lors = np.load(lor_file)[lor_start:lor_end, :]
  return image / effmap + np.sum(lors) / image.size


class ReconstructionService(recon_pb2_grpc.ReconstructionServicer):
  def ReconStep(self, request, context):
    result = recon_pb2.Image()
    image_shape = list(request.image_shape)
    result_image = reconstruct_step(
        request.efficiency_map_file, request.lor_file, request.lor_start,
        request.lor_end,
        np.array(request.image).reshape(image_shape))
    result.values.extend(result_image.flatten())
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
