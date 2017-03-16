import os
import subprocess
from ctypes import cdll
import time

def handler(event, context):
    results = []
    vecStrAnswer = []
    strServerAddress = "http://github.com"

    commandWRK = "./wrk -t1 -c1 -d5s '"+ strServerAddress+"'"
    commandPyresttest = "python pyresttest/resttest.py https://api.github.com github_api_test.yaml"

    command = commandWRK
    print(command)
    try:
        output_path = subprocess.check_output(command, shell=True,stderr=subprocess.STDOUT)
        print(output_path)
    except subprocess.CalledProcessError as e:
        print(e.output)

    command = commandPyresttest
    print(command)
    try:
        output_path = subprocess.check_output(command, shell=True,stderr=subprocess.STDOUT)
        print(output_path)
    except subprocess.CalledProcessError as e:
        print(e.output)