from __future__ import print_function

import urllib
import os
import subprocess
import boto3


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(SCRIPT_DIR, 'lib')

def handler(event, context):
    try:
        imgfilepath = '/tmp/imgres.png'
        jsonfilepath = '/tmp/result'
        urllib.urlretrieve("https://www.dropbox.com/s/zol1inds6s3bs5a/imgres.png?dl=1", imgfilepath)
        command = 'LD_LIBRARY_PATH={} TESSDATA_PREFIX={} ./tesseract {} {}'.format(
            LIB_DIR,
            SCRIPT_DIR,
            imgfilepath,
            jsonfilepath,
        )
        print(command)

        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            f = open(jsonfilepath+'.txt')
            print(f.read())
        except subprocess.CalledProcessError as e:
            print(e.output)
    except Exception as e:
        print(e)
        raise e
    return 0 