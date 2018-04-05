from .utils import MainAxis
from ...core import Model


class Rotation:
    def __init__(self, main_axis: MainAxis):
        if isinstance(main_axis, str):
            self.main_axis = MainAxis(main_axis.lower())
        elif isinstance(main_axis, MainAxis):
            self.main_axis = main_axis
        else:
            raise TypeError("Invalid main_axis: {}.".format(main_axis))

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def rotate_list_forward(self, l):
        return [l[i] for i in self.forward()]


class RotationImage(Rotation):
    def forward(self):
        return {
            'x': [2, 0, 1],
            'y': [1, 0, 2],
            'z': [0, 1, 2]
        }[self.main_axis.value]

    def backward(self):
        return {
            'x': [1, 2, 0],
            'y': [1, 0, 2],
            'z': [0, 1, 2]
        }[self.main_axis.value]


def rotate_list_image(main_axis, l):
    return RotationImage(main_axis).rotate_list_forward(l)


class RotationLORs(Rotation):
    def forward(cls):
        return {
            'x': [1, 2, 0],
            'y': [0, 2, 1],
            'z': [0, 1, 2]
        }[self.main_axis.value]

    def backward(cls):
        return {
            'x': [2, 0, 1],
            'y': [0, 2, 1],
            'z': [0, 1, 2]
        }[self.main_axis.value]
