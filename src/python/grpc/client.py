import numpy as np

import grpc
import recon_pb2
import recon_pb2_grpc

<<<<<<< HEAD

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
=======
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

>>>>>>> 69901e0fe34721a58822a0dddaeee48b5309f8e7

if __name__ == "__main__":
  main()