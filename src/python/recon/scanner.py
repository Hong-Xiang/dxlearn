import numpy as np
from typing import Iterable
from dxl.shape.rotation.matrix import *
from dxl.shape.utils.vector import Vector3
from dxl.shape.utils.axes import Axis3, AXIS3_X, AXIS3_Z

from computeMap import TorMap, SiddonMap

from dxl.learn.preprocess import preprocess
import click
import time

import itertools

root = '/home/chengaoyu/code/Python/gitRepository/dxlearn/develop-cgy/maps_tor_2m/'

class Vec3():
  def __init__(self, x=0, y=0, z=0):
    self._x = x
    self._y = y
    self._z = z

  @property
  def x(self):
    return self._x

  @property
  def y(self):
    return self._y

  @property
  def z(self):
    return self._z

  @property
  def value(self):
    return np.array([self.x, self.y, self.z])


class Block():
  def __init__(self, block_size: Vec3, center: Vec3, grid: Vec3,
               rad_z: np.float32):
    self._block_size = block_size
    self._center = center
    self._grid = grid
    self._rad_z = rad_z

  @property
  def grid(self):
    return self._grid

  @property
  def center(self):
    return self._center

  @property
  def rad_z(self):
    return self._rad_z

  @property
  def block_size(self):
    return self._block_size

  def meshes(self) -> np.array:
    """
        return all of the crystal centers in a block
        """
    bottom_p = -self.block_size.value / 2 + self.center.value
    mesh_size = self._block_size.value / self.grid.value
    meshes = np.zeros([self.grid.z, self.grid.y, self.grid.x, 3])
    grid = self.grid
    # print(bottom_p)
    # print()

    for ix in range(grid.x):
      meshes[:, :, ix, 0] = (ix + 0.5) * mesh_size[0] + bottom_p[0]
    for iy in range(grid.y):
      meshes[:, iy, :, 1] = (iy + 0.5) * mesh_size[1] + bottom_p[1]
    for iz in range(grid.z):
      meshes[iz, :, :, 2] = (iz + 0.5) * mesh_size[2] + bottom_p[2]
    # print(meshes.shape)
    meshes = np.transpose(meshes)
    source_axis = AXIS3_X
    target_axis = Axis3(Vector3([np.cos(self.rad_z), np.sin(self.rad_z), 0]))
    rot = axis_to_axis(source_axis, target_axis)

    rps = rot @ np.reshape(meshes, (3, -1))
    return np.transpose(rps)


class RingPET():
  def __init__(self, inner_radius: np.float32, outer_radius: np.float32,
               gap: np.float32, num_rings: np.int32, num_blocks: np.int32,
               block_size: Vec3, grid: Vec3):
    self._inner_radius = inner_radius
    self._outer_radius = outer_radius
    self._num_rings = num_rings
    self._num_blocks = num_blocks
    self._block_size = block_size
    self._grid = grid
    self._gap = gap
    # self._block_list: Iterable[Block] = self._make_blocks()
    self._rings = self._make_rings()

  @property
  def inner_radius(self):
    return self._inner_radius

  @property
  def outer_radius(self):
    return self._outer_radius

  @property
  def num_blocks(self):
    return self._num_blocks

  @property
  def num_rings(self):
    return self._num_rings

  @property
  def block_size(self):
    return self._block_size

  @property
  def grid(self):
    return self._grid

  @property
  def gap(self):
    return self._gap

  # @property
  # def block_list(self):
  #     return self._block_list

  # @property
  def rings(self, num: np.int32):
    """
        obtain a block list of a single ring 
        0: the bottom one
        num_rings - 1: the top one 
        """
    return self._rings[num]

  def _make_rings(self):
    num_rings = self.num_rings
    num_blocks = self.num_blocks
    block_size = self.block_size
    grid = self.grid
    gap = self.gap
    ri = self.inner_radius
    ro = self.outer_radius

    rings = []
    bottom_z = -(block_size.z + gap) * (num_rings - 1) / 2
    block_x_offset = (ri + ro) / 2
    for ir in range(num_rings):
      block_z_offset = bottom_z + ir * (block_size.z + gap)
      pos = Vec3(block_x_offset, 0, block_z_offset)
      # print(num_blocks)
      block_list: Iterable[Block] = []
      for ib in range(num_blocks):
        phi = 360.0 / num_blocks * ib
        rad_z = phi / 180 * np.pi
        block_list.append(Block(block_size, pos, grid, rad_z))
      rings.append(block_list)
    # print(len(rings))
    return rings

  def wash_lors(self, lors):
    """
    to process the list mode data with specific DOI information
    """
    
    pass

  # def _make_blocks(self):
  #     num_rings = self.num_rings
  #     num_blocks = self.num_blocks
  #     block_size = self.block_size
  #     grid = self.grid
  #     gap = self.gap
  #     ri = self.inner_radius
  #     ro = self.outer_radius
  #     block_list: Iterable[Block] = []
  #     bottom_z = -(block_size.z + gap)*(num_rings-1)/2
  #     block_x_offset = (ri + ro)/2
  #     for ir in range(num_rings):
  #         block_z_offset = bottom_z + ir*(block_size.z + gap)
  #         pos = Vec3(block_x_offset, 0, block_z_offset)
  #         for ib in range(num_blocks):
  #             phi = 360.0/num_blocks*ib
  #             rad_z = phi/180*np.pi
  #             block_list.append(Block(block_size, pos, grid, rad_z))
  #     return block_list


# class BlockList():


def print_block_pair(bps):
  def pb(b):
    return "({},{})".format(b.center.x, b.center.y)

  return "<{}|{}>".format(pb(bps[0]), pb(bps[1]))


def make_block_pairs(block_list):
  """
    return the block pairs in a block list.
    """
  block_pairs = []
  if len(block_list) == 1:
    # print("1")
    ring = block_list[0]
    # print('len ring:', len(ring))

    block_pairs = [[b1, b2] for i1, b1 in enumerate(ring)
                   for i2, b2 in enumerate(ring) if i1 < i2]
    # print('len bps:', len(block_pairs))
    # msg = [print_block_pair(bps) for bps in block_pairs]
    # for bps in block_pairs:
    # print(bps)
    # print('\n'.join(msg))
  else:
    # print("2")
    ring1 = block_list[0]
    ring2 = block_list[1]
    block_pairs = [[b1, b2] for b1 in ring1 for b2 in ring2]
  # print(block_pairs)
  return block_pairs


def make_lors(block_pairs):
    lors = []
    # print((block_pairs))
    for ibp in block_pairs:
        b0 = ibp[0]
        b1 = ibp[1]
        m0 = b0.meshes()
        m1 = b1.meshes()
        lors.append(list(itertools.product(m0, m1)))
    return np.array(lors).reshape(-1, 6)

def make_maps(start, end):
    # rpet = RingPET(400.0, 420.0, 0.0, 432, 20, Vec3(20, 122.4, 3.4), Vec3(5, 36, 1))
    # rpet = RingPET(400, 420, 0.0, 540, 48, Vec3(20, 51.3, 3.42),Vec3(1, 15, 1))

    # rpet = RingPET(400.0, 420.0, 0.0, 416, 48, Vec3(20., 51.3, 3.42), Vec3(5, 15, 1))
    rpet = RingPET(400.0, 420.0, 0.0, 540, 48, Vec3(20., 51.3, 3.42), Vec3(1, 15, 1))
    r1 = rpet.rings(0)
    total_time = 0
    for ir in range(start, end):
        print("start compute the {} th map.".format(ir))
        st = time.time()
        # ir = 6
        r2 = rpet.rings(ir)
        bs = make_block_pairs([r1,r2])
        lors = make_lors(bs)
        xlors, ylors, zlors = preprocess(lors)  
        xlors = xlors[:, [1, 2, 0, 4, 5, 3]] # y z x
        ylors = ylors[:, [0, 2, 1, 3, 5, 4]] # x z y
        grid = [540, 195, 195]  
        # origin = [-711.36, -333.45, -333.45]
        center = [0., 0., 0.]
        size = [1846.8, 666.9, 666.9]
        
        # voxsize = [3.42, 3.42, 3.42]
        # et = time.time()
        # subnum = 3
        # xsub = xlors.shape[0]//3
        # ysub = ylors.shape[0]//3
        # zsub = zlors.shape[0]//3
        # print("lorshape:", xlors.shape, ylors.shape, zlors.shape)
        # subxlors = [xlors[0 : xsub, :], xlors[xsub:xsub*2, :], xlors[xsub*2:, :]]
        # subylors = [ylors[0 : ysub, :], ylors[ysub:ysub*2, :], ylors[ysub*2:, :]]
        # subzlors = [zlors[0 : zsub, :], zlors[zsub:zsub*2, :], zlors[zsub*2:, :]]
        # # print("subxlors shape:", subxlors[0].shape)
        # effmap = []
        # for isub in range(subnum): 
        #     effmap.append(computeMap(grid, center, size, subxlors[isub], subylors[isub], subzlors[isub]))
        # summap = effmap[0] + effmap[1] +effmap[2]
        
        # effmap = SiddonMap(grid, voxsize, origin, lors)

        effmap = TorMap(grid, center, size, xlors, ylors, zlors)
        np.save(root+'effmap_{}.npy'.format(ir), effmap)
        et = time.time()
        tdiff = et-st
        print("{} th map use: {} seconds".format(ir, tdiff))
        total_time += tdiff
        print("total time: {} seconds".format(total_time))
        print("time remain: {} seconds".format(total_time/(ir-start + 1)*(end - ir - 1)))
        
# def merge_effmap(num_rings):
#   """
#   to do: implemented in GPU to reduce the calculated time
#   """
#   temp = np.load(root+'effmap_{}.npy'.format(0))
#   final_map = np.zeros(temp.shape)
#   print(final_map.shape)
#   st = time.time()
#   for ir in range(num_rings):
#     temp = np.load(root+'effmap_{}.npy'.format(ir))
#     print("process :{}/{}".format(ir+1, num_rings))
#     for jr in range(num_rings - ir):
#       if ir == 0:
#         final_map[:,:,jr:num_rings] += temp[:,:,0:num_rings-jr]/2
#       else:
#         final_map[:,:,jr:num_rings] += temp[:,:,0:num_rings-jr]
#     et = time.time()
#     tr = (et -st)/(num_rings*(num_rings-1)/2 - (num_rings - ir - 1)*(num_rings-ir-2)/2)*((num_rings - ir-1)*(num_rings-ir-2)/2)
#     print("estimated time remains: {} seconds".format(tr))
#   np.save(root+'summap.npy', final_map)

def test_tor_map():
  rpet = RingPET(400.0, 420.0, 0.0, 432, 20, Vec3(20, 122.4, 3.4), Vec3(5, 36, 1))
  # rpet = RingPET(400.0, 400.0, 0.0, 400, 20, Vec3(20, 160, 4), Vec3(1, 16, 1))
  r1 = rpet.rings(num=0)
  r2 = rpet.rings(num=216 + 50)
  bs = make_block_pairs([
      r1,
  ])
  lors = make_lors(bs)
  # exit()
  nb_lors = len(lors)
  # np.save('./debug/lors_{}.npy'.format(nb_lors), lors)
  print('[INFO :: DXL.LEARN] Number of lors:', len(lors))
  grid = [160, 160, 440]
  center = [0., 0., 0.]
  # size = [544.*2., 544.*3., 544.*4.]
  size = [544., 544., 1496.]
  # volsize = [7., 7., 7.]

  def kernel(lors):
    # exit()
    xlors, ylors, zlors = preprocess(lors)
    xlors = xlors[:, [1, 2, 0, 4, 5, 3]]  # y z x
    ylors = ylors[:, [0, 2, 1, 3, 5, 4]]  # x z y
    np.save('./debug/xlors.npy', xlors)
    return computeMap(grid, center, size, xlors, ylors, zlors)

  st = time.time()
  # stack_shape = [nb_lors] + grid[:-1]
  # print(stack_shape)
  # result = np.zeros(stack_shape)
  # import tensorflow as tf  
  start_time_it = time.time()
  # with tf.Session() as sess:
  seperate = False
  if seperate:
    for i in range(nb_lors):
      effmap = kernel(lors[i:i+1, :])
      img_sz = np.sum(np.sum(effmap, axis=0), axis=0)
      zslice = np.argmax(img_sz)
      result[i, :] = effmap[:,:,zslice]
      itime = (time.time()-start_time_it)/(i+1)
      print('[INFO :: DXL.LEARN] Running LOR {} of {}, {:0.3f} s/step, remain {:0.3f} s.'.format(i, nb_lors, itime, itime*(nb_lors-i)))
  else:
    effmap = kernel(lors)
    np.save('./debug/effmap_{}.npy'.format(0), effmap)
  # np.save('./debug/effmap_stack_{}.npy'.format(nb_lors), result)
  et = time.time()
  tdiff = et - st
  print(effmap)
  print("the total time: {} seconds".format(tdiff))


def test_siddon_map():
  rpet = RingPET(400.0, 420.0, 0.0, 432, 4, Vec3(20, 122.4, 3.4), Vec3(1, 4, 1))
  rpet = RingPET(400.0, 420.0, 0.0, 416, 48, Vec3(20, 51.3, 3.42), Vec3(5, 15, 1))
  # rpet = RingPET(400.0, 400.0, 0.0, 400, 4, Vec3(20, 160, 4), Vec3(1, 1, 1))
  r1 = rpet.rings(num=215)
  r2 = rpet.rings(num=216 + 50)
  bs = make_block_pairs([
      r1,
  ])
  # print(len(bs))
  lors = make_lors(bs)
  # print(lors)
  # np.save('./debug/lors.npy', lors[:1000, :])
  print(len(lors))
  # exit()
  xlors, ylors, zlors = preprocess(lors)
  # np.save('./debug/xlors.npy', xlors[:1000, :])
  # np.save('./debug/ylors.npy', ylors[:1000, :])
  # np.save('./debug/zlors.npy', zlors[:1000, :])
  xlors = xlors[:, [1, 2, 0, 4, 5, 3]]  # y z x
  ylors = ylors[:, [0, 2, 1, 3, 5, 4]]  # x z y
  # exit()
  grid = [160, 240, 320]
  center = [0., 0., 0.]
  # size = [544.*2., 544.*3., 544.*4.]
  size = [1120., 1680., 2240.]
  origin = [-544., -840., -1120.]
  volsize = [7., 7., 7.]
  st = time.time()
  # print(xlors.shape)
  # print(ylors.shape)
  slors = np.hstack((lors, np.zeros((lors.shape[0], 1))))
  print(slors.shape)
  # exit()
  effmap = siddonMap(grid, volsize, origin, slors)
  # effmap = computeMap(grid, center, size, xlors, ylors, zlors)
  np.save('./debug/effmap_{}.npy'.format(0), effmap)
  et = time.time()
  tdiff = et - st
  print(effmap)
  print("the total time: {} seconds".format(tdiff))





def main(start:int, end:int):
    make_maps(start, end)

@click.command()
@click.option('--start', '-s', help = 'start ring', type = int)
@click.option('--end',   '-e', help = 'end ring', type = int)
# @click.option('--task', '-t', help = 'task', type = int, default = 0)
def cli(start, end):
    main(start, end)
    # merge_maps()

if __name__ == "__main__":
    cli()
    # merge_effmap(540, './maps2/')