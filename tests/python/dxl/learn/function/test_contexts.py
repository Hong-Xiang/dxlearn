from dxl.learn import dependencies
import tensorflow as tf

def test_tensorflow_control_dependencies(tensorflow_test_session):
    a = tf.get_variable("x", initializer=0.0)
    b = a.assign(1.0)
    with dependencies([b]):
        o = tf.no_op()
    tensorflow_test_session.run(tf.global_variables_initializer())
    tensorflow_test_session.run(o)
    assert tensorflow_test_session.run(a) == 1.0
    
    
