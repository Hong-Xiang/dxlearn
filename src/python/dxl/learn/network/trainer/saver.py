import tensorflow as tf
from pathlib import Path
from ..graph import Graph
import arrow
from ..utils import logger
from ..session import ThisSession


class Saver(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            MODEL_DIR = 'model_dir'
            CHECKPOINT_NAME = 'checkpoint_name'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            SAVER = 'saver'

    def __init__(self,
                 info='saver',
                 variables=None,
                 config=None,
                 *,
                 model_dir=None,
                 ckpt_name=None,
                 save_interal=None):
        super().__init__(info, tensors=variables)

    def kernel(self):
        self.graphs[self.KEYS.SUBGRAPH.SAVER] = tf.train.Saver(variables)

    def checkpoint_path(self, step=None):
        """
        Return full path of checkpoint directory, if step is not `None`,
        checkpoint with step will be returned.
        """
        return Path(
            self.config(self.KEYS.CONFIG.MODEL_DIR) / self.config(
                self.KEYS.CONFIG.CHECKPOINT_NAME))

    def save(self, session=None):
        logger.info('Save model to {}.'.format(self.config()))

    def load(self, step=None, session=None):
        if session is None:
            session = ThisSession.session().data
