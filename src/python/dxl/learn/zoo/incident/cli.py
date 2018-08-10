import click

# from dxl.learn.zoo.incident.model import test_model
from dxl.core.debug import enter_debug


@click.group()
# @click.option('--debug', '-d', help='enter debug if ')
def incident():
    enter_debug()


# from .model import test_model
# incident.add_command(test_model)
from .train import train
incident.add_command(train)
# from ._keras import train_keras
# incident.add_command(train_keras)

from .grid_search import gridc
incident.add_command(gridc)