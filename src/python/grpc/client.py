import numpy as np

import grpc
import recon_pb2
import recon_pb2_grpc

MAX_MESSAGE_LENGTH = 20000000

def main():
  channel = grpc.insecure_channel(
      'localhost:50050',
      options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
  stub = recon_pb2_grpc.ReconstructionStub(channel)
  effmap_file = './debug/map.npy'
  lor_files = ['./debug/{}lors.npy'.format(a) for a in ['x', 'y', 'z']]
  lor_range = [0, int(1e5)]
  grid = [150, 150, 150]
  center = [0., 0., 0.]
  size = [150., 150., 150.]
  image = np.ones(grid)
  req = recon_pb2.ReconPayload()
  req.efficiency_map_file = effmap_file
  req.lor_files.extend(lor_files)
  req.lor_range.extend(lor_range)
  req.image.extend(image.flatten())
  req.grid.extend(grid)
  req.center.extend(center)
  req.size.extend(size)
  res = stub.ReconStep(req)
  result = np.array(res.values).reshape(grid)
  print(result)
  np.save('rpc_result.npy', result)


if __name__ == "__main__":
  main()