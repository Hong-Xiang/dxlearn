# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from fs import path as fp
from .blocks import Conv2D, StackedConv2D, InceptionBlock, UnitBlock
from ...core import Model
from ...core import Tensor

__all__ = [
    'ResidualIncept',
    'ResidualStackedConv',
    'StackedResidualIncept',
    'StackedResidualConv'
]


class ResidualIncept(Model):
    """ResidualIncept Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        ratio: The decimal.
        sub_block: InceptionBlock instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass
        class CONFIG:
            RATIO = 'ratio'
            SUB_BLOCK = 'sub_block'

    def __init__(self, name,
                 input_tensor=None,
                 ratio=0.3,
                 sub_block: InceptionBlock=None,
                 graph_info=None):       
        super().__init__(
            name,
            inputs={
                self.KEYS.TENSOR.INPUT: input_tensor
            },
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.RATIO: ratio,
                self.KEYS.CONFIG.SUB_BLOCK: sub_block
            })
    
    def pre_kernel(self, inputs, is_create):
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        if not isinstance(sub_block, (UnitBlock, InceptionBlock)):
            sub_block = InceptionBlock(
                name='incept',
                input_tensor=inputs[self.KEYS.TENSOR.INPUT],
                paths=3,
                activation='incept')
            self.update_config(self.KEYS.CONFIG.SUB_BLOCK, sub_block)

        if is_create:
            for k, v in inputs.items():
                self.inputs[k] = v
        if isinstance(inputs, (Tensor, tf.Tensor)):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        if inputs is not None:
            if isinstance(inputs, dict):
                for k in self.inputs:
                    if not k in inputs:
                        inputs[k] = self.inputs[k]
        return inputs

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        h = sub_block(inputs)
        with tf.name_scope('add'):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x
    

class ResidualStackedConv(Model):
    """ ResidualStackedConv Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        ratio: The decimal.
        sub_block: StackedConv2D instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass
        class CONFIG:
            RATIO = 'ratio'
            SUB_BLOCK = 'sub_block'

    def __init__(self, name,
                 input_tensor=None,
                 ratio=0.1,
                 sub_block: StackedConv2D=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs={
                self.KEYS.TENSOR.INPUT: input_tensor
            },   
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.RATIO: ratio,
                self.KEYS.CONFIG.SUB_BLOCK: sub_block
            })

    def pre_kernel(self, inputs, is_create):
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        if not isinstance(sub_block, (UnitBlock, StackedConv2D)):
            sub_block = StackedConv2D(
                name='conv',
                input_tensor=inputs[self.KEYS.TENSOR.INPUT],
                nb_layers=2,
                filters=1,
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='basic')
            self.update_config(self.KEYS.CONFIG.SUB_BLOCK, sub_block)

        if is_create:
            for k, v in inputs.items():
                self.inputs[k] = v
        if isinstance(inputs, (Tensor, tf.Tensor)):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        if inputs is not None:
            if isinstance(inputs, dict):
                for k in self.inputs:
                    if not k in inputs:
                        inputs[k] = self.inputs[k]
        return inputs
        
    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        h = sub_block(inputs)
        with tf.name_scope('add'):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x
    

class StackedResidualIncept(Model):
    """StackedResidual Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        nb_layers: Integer.
        sub_block: ResidualIncept Instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass
        class CONFIG:
            NB_LAYERS = 'nb_layers'
            SUB_BLOCK = 'sub_block'
    
    def __init__(self, name,
                 input_tensor=None,
                 nb_layers=None,
                 sub_block: ResidualIncept=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs={
                self.KEYS.TENSOR.INPUT: input_tensor
            },
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.NB_LAYERS: nb_layers,
                self.KEYS.CONFIG.SUB_BLOCK: sub_block
            })

    def pre_kernel(self, inputs, is_create):
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        if not isinstance(sub_block, (UnitBlock, ResidualIncept)):
            sub_block = ResidualIncept(
                name='incept',
                input_tensor=inputs[self.KEYS.TENSOR.INPUT],
                ratio=0.3)
            self.update_config(self.KEYS.CONFIG.SUB_BLOCK, sub_block)

        if is_create:
            for k, v in inputs.items():
                self.inputs[k] = v
        if isinstance(inputs, (Tensor, tf.Tensor)):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        if inputs is not None:
            if isinstance(inputs, dict):
                for k in self.inputs:
                    if not k in inputs:
                        inputs[k] = self.inputs[k]
        return inputs
    
    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            x = sub_block(inputs)
        return x


class StackedResidualConv(Model):
    """StackedResidual Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        nb_layers: Integer.
        sub_block: ResidualStackedConv Instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass
        class CONFIG:
            NB_LAYERS = 'nb_layers'
            SUB_BLOCK = 'sub_block'
    
    def __init__(self, name,
                 input_tensor=None,
                 nb_layers=None,
                 sub_block: ResidualStackedConv=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs={
                self.KEYS.TENSOR.INPUT: input_tensor
            },
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.NB_LAYERS: nb_layers,
                self.KEYS.CONFIG.SUB_BLOCK: sub_block
            })

    def pre_kernel(self, inputs, is_create):
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        if not isinstance(sub_block, (UnitBlock, ResidualStackedConv)):
            sub_block = ResidualStackedConv(
                name='stacked_conv',
                input_tensor=inputs[self.KEYS.TENSOR.INPUT],
                ratio=0.1)
            self.update_config(self.KEYS.CONFIG.SUB_BLOCK, sub_block)

        if is_create:
            for k, v in inputs.items():
                self.inputs[k] = v
        if isinstance(inputs, (Tensor, tf.Tensor)):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        if inputs is not None:
            if isinstance(inputs, dict):
                for k in self.inputs:
                    if not k in inputs:
                        inputs[k] = self.inputs[k]
        return inputs
    
    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            x = sub_block(inputs)
        return x
    

