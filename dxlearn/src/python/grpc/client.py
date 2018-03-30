import numpy as np

import grpc
import recon_pb2
import recon_pb2_grpc


def main():
  channel = grpc.insecure_channel('localhost:50050')
  stub = recon_pb2_grpc.ReconstructionStub(channel)
  effmap_file = './effmap.npy'
  lor_file = './lor.npy'
  lor_start = 0
  lor_end = 10
  image_shape = [10, 10, 10]
  image = np.ones(image_shape)
  req = recon_pb2.ReconPayload()
  req.efficiency_map_file = effmap_file
  req.lor_file = lor_file
  req.lor_start = lor_start
  req.lor_end = lor_end
  req.image.extend(image.flatten())
  req.image_shape.extend(image_shape)
  res = stub.ReconStep(req)
  result = np.array(res.values).reshape(image_shape)
  print(result)

if __name__ == "__main__":
  main()