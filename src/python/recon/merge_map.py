import numpy as np
import time

def merge_effmap(num_rings, file_dir):
  """
  to do: implemented in GPU to reduce the calculated time
  """
  root = '/home/chengaoyu/code/Python/gitRepository/dxlearn/develop-cgy/'
  temp = np.load(root+file_dir+'effmap_{}.npy'.format(0))
  final_map = np.zeros(temp.shape).transpose()
  print(final_map.shape)
  st = time.time()
  for ir in range(num_rings):
    temp = np.load(root+file_dir+'effmap_{}.npy'.format(ir)).transpose()
    print("process :{}/{}".format(ir+1, num_rings))
    for jr in range(num_rings - ir):
      if ir == 0:
        final_map[jr:num_rings,:,:] += temp[0:num_rings-jr,:,:]/2
      else:
        final_map[jr:num_rings,:,:] += temp[0:num_rings-jr,:,:]
    et = time.time()
    tr = (et -st)/(num_rings*(num_rings-1)/2 - (num_rings - ir - 1)*(num_rings-ir-2)/2)*((num_rings - ir-1)*(num_rings-ir-2)/2)
    print("estimated time remains: {} seconds".format(tr))
  # odd = np.arange(0, num_rings, 2)
  # even = np.arange(1, num_rings, 2)
  # final_map = final_map[:,:,odd] + final_map[:,:, even]

  np.save(root+file_dir+'summap.npy', final_map)

if __name__ == '__main__':
    merge_effmap(540, 'maps_tor_2m/')
