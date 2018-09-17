from dxl.learn.model.dropout import *
from dxl.learn.model.dnns import Dense
from doufo.tensor.binary import all_close


def test_dropout_basic(clean_config):
    x = tf.ones([2, 2], dtype=tf.float32)
    d = DropOut(keep_prob=0.0001)
    m1 = Dense('d1', 2)
    x = d(x)
    res = m1(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        temp = []
        for i in range(10000):
            temp.append(sess.run(res))
        res = [1 for i in temp if not all_close(i, [[0, 0], [0, 0]])]
        prob = len(res) / 1000
        print(prob)
    assert prob < 0.1
