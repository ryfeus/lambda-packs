# -*- coding: utf-8 -*-
from skimage import io
import urllib
import skimage.segmentation as segmentation

def handler(event, context):
    urllib.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/3/38/JPEG_example_JPG_RIP_001.jpg", "/tmp/hi.jpg")
    img = io.imread('/tmp/hi.jpg')    
    print(img)
    return 0
