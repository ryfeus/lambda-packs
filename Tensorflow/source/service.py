
import tensorflow as tf

def handler(event, context):
    a = tf.constant(1)
    b = tf.constant(2)
    with tf.Session() as sess:
        return str(sess.run(a + b))
