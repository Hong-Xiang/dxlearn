# from dxl.learn.network.saver import Saver
from dxl.learn.test import TestCase
import tensorflow as tf
import tempfile
import pytest

@pytest.mark.skip('niy')
class TestSaver(TestCase):
    def test_save_default_variables(self):
        x = tf.get_variable('x', [], tf.float32)
        xa = x.assign(1.0)
        with tempfile.TemporaryDirectory() as dirname:
            saver = Saver(dirname)
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(xa)
                assert sess.run(x) == 1.0
                saver.save(dirname)
            with self.test_session() as sess:
                with pytest.raises(TypeError):
                    sess.run(x)
            with self.test_session() as sess:
                saver.load()
                assert sess.run(x) == 1.0
