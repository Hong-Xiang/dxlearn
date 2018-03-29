import numpy as np
from typing import Iterable
from dxl.shape.rotation.matrix import *
from dxl.shape.utils.vector import Vector3
from dxl.shape.utils.axes import Axis3, AXIS3_X, AXIS3_Z

import itertools

import time
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
    def __init__(self, block_size: Vec3, center: Vec3, grid: Vec3, rad_z: np.float32):
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

    def meshes(self) ->np.array:
        """
        return all of the crystal centers in a block
        """
        bottom_p = -self.block_size.value/2 + self.center.value
        mesh_size = self._block_size.value/self.grid.value
        meshes = np.zeros([self.grid.z, self.grid.y, self.grid.x, 3])
        grid = self.grid
        

        for ix in range(grid.x):
            meshes[:, :, ix, 0] = (ix+0.5)*mesh_size[0] + bottom_p[0]
        for iy in range(grid.y):
            meshes[:, iy, :, 1] = (iy+0.5)*mesh_size[1] + bottom_p[1]
        for iz in range(grid.z):
            meshes[iz, :, :, 2] = (iz+0.5)*mesh_size[2] + bottom_p[2]
        # print(meshes.shape)
        meshes = np.transpose(meshes)
        source_axis = AXIS3_X
        target_axis = Axis3(Vector3([np.cos(self.rad_z), np.sin(self.rad_z), 0]))
        rot = axis_to_axis(source_axis, target_axis)
        
        rps = rot@np.reshape(meshes, (3, -1))
        return np.transpose(rps)

        


class RingPET():
    def __init__(self, inner_radius: np.float32, outer_radius: np.float32, gap: np.float32,
                 num_rings: np.int32, num_blocks: np.int32, block_size: Vec3, grid: Vec3):
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._num_rings = num_rings
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._grid = grid
        self._gap = gap
        self._block_list: Iterable[Block] = self._make_blocks()

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
    
    @property
    def block_list(self):
        return self._block_list

    def _make_blocks(self):
        num_rings = self.num_rings
        num_blocks = self.num_blocks
        block_size = self.block_size
        grid = self.grid
        gap = self.gap
        ri = self.inner_radius
        ro = self.outer_radius
        block_list: Iterable[Block] = []
        bottom_z = -(block_size.z + gap)*(num_rings-1)/2
        block_x_offset = (ri + ro)/2
        for ir in range(num_rings):
            block_z_offset = bottom_z + ir*(block_size.z + gap)
            pos = Vec3(block_x_offset, 0, block_z_offset)
            for ib in range(num_blocks):
                phi = 360.0/num_blocks*ib
                rad_z = phi/180*np.pi
                block_list.append(Block(block_size, pos, grid, rad_z))
        return block_list

    def blockpairs(self):
        """
        return the block pairs list in a ring scanner
        """
        block_pairs = []
        for b1 in self.block_list:
            for b2 in self.block_list:
                if abs(b1.rad_z - b2.rad_z) > 1e-6:
                    block_pairs.append([b1,b2])
        return block_pairs

def make_lors(blockpairs):
    lors = []
    for ibp in blockpairs:
        b0 = ibp[0]
        b1 = ibp[1]
        m0 = b0.meshes()
        m1 = b1.meshes()
        lors.append(list(itertools.product(m0, m1)))
    return np.array(lors, dtype = np.float32).reshape((-1, 6))

if __name__ == '__main__':
    # rpet = RingPET(400.0, 420.0, 0.0, 9, 20, Vec3(20, 68.0, 163.2), Vec3(1,20,48))
    # print(rpet)
    start_time = time.time()
    b0 = Block(Vec3(1,1,1),Vec3(0,0,3),Vec3(1,100,100), np.pi)
    b1 = Block(Vec3(1,1,1),Vec3(0,0,3),Vec3(1,100,100), 0)    
    bps = [[b0, b1],]
    m = make_lors(bps)
    print(m)
    print(m.shape)
    end_time = time.time()
    time_diff = end_time - start_time
    print("time cost: {} seconds".format(time_diff))
    # bps = rpet.blockpairs()
    # b1 = bps
    # print(bps[0][0])
    # print(bps[0][1])    
    # print(len(bps))

