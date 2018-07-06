from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import boto3
import subprocess
import numpy as np
import tensorflow as tf
# import keras
import time 
import argparse
import os.path
import re
import sys
import urllib
import StringIO
import io

SESSION = None

def downloadFromS3(strBucket,strKey,strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)

def getObject(strBucket,strKey):
    s3_client = boto3.client('s3')
    s3_response_object = s3_client.get_object(Bucket=strBucket, Key=strKey)
    return s3_response_object['Body'].read()  
    # return StringIO.StringIO(object_content)
    # return io.BytesIO(object_content)

def create_graph():
    # with tf.gfile.FastGFile(os.path.join('/tmp/imagenet/', 'classify_image_graph_def.pb'), 'rb') as f:
    # with tf.gfile.FastGFile(getObject('ryfeuslambda','tensorflow/imagenet/classify_image_graph_def.pb'), 'rb') as f:
    # with open(getObject('ryfeuslambda','tensorflow/imagenet/classify_image_graph_def.pb'), 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     _ = tf.import_graph_def(graph_def, name='')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(getObject('ryfeuslambda','tensorflow/imagenet/classify_image_graph_def.pb'))
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
    global SESSION
    loaded = False
    load_time = 0
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()  
    if SESSION is None:
        loaded = True
        start = time.time()
        SESSION = tf.InteractiveSession()
        create_graph()
        end = time.time()
        load_time = end - start
    softmax_tensor = tf.get_default_graph().get_tensor_by_name('softmax:0')
    start = time.time()
    predictions = SESSION.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    end = time.time()
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-5:][::-1]
    return str((top_k,loaded,round(load_time,2),round(end-start,2)))

def handler(event, context):
    if not os.path.exists('/tmp/imagenet/'):
        os.makedirs('/tmp/imagenet/')
    downloadFromS3('ryfeuslambda','tensorflow/imagenet/classify_image_graph_def.pb','/tmp/imagenet/classify_image_graph_def.pb')

    strFile = '/tmp/imagenet/inputimage.jpg'
    if ('imagelink' in event):
        urllib.urlretrieve(event['imagelink'], strFile)
    else:
        downloadFromS3('ryfeuslambda','tensorflow/imagenet/cropped_panda.jpg',strFile)

    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
    )
    parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    image = os.path.join('/tmp/imagenet/', 'inputimage.jpg')
    strResult = run_inference_on_image(image)
    return strResult
