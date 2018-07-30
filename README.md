# lambda-packs 

Precompiled packages for AWS lambda

## How to start

1. https://aws.amazon.com/lambda/ and create/sign in into account
2. Lambda > Functions - Create lambda function
3. Blank function
4. Configure triggers - Next
5. Configure function
  - Runtime - Python 2.7
6. Lambda function handler and role
  - Handler - service.handler
  - Role - Create new role from template(s)
  - Role name - test
  - Policy templates - Simple Microservice Permissions
7. Advanced settings
  - Memory (MB) 128
  - Timeout 1 min 0 sec
6. Code entry type - Upload a .ZIP file - choose Pack.zip from rep
7. Test -> Save and test

## How to modify

1. Modify service.py file from sources folder
2. Choose all files in sources folder when compressing, don't put it in one folder
3. Upload zip file on function page

## Current packs

### Selenium PhantomJS

#### Intro

Selenium on PhantomJS. In fact - a ready-made tool for web scraping. For example, the demo now opens a random page in Wikipedia and sends its header. (PhantomJS at the same time disguises itself as a normal browser, knows how to log in, click and fill out forms) Also added requests, so you can do API requests for different resources to discard / take away the information.

Useful for web testing and scraping.

#### Demo

Current demo opens random page from wiki (https://en.wikipedia.org/wiki/Special:Random) and prints title.

#### Serverless start

```
serverless install -u https://github.com/ryfeus/lambda-packs/tree/master/Selenium_PhantomJS/source -n selenium-requests
cd selenium-requests
serverless deploy
serverless invoke --function main --log

You can also see the results from the API Gateway endpoint in a web browser.

#### Documentation

https://selenium-python.readthedocs.io/



---
### Pyresttest + WRK

#### Intro

What does the lambda have to do with it? In a nutshell on AWS in one region you can simultaneously run 200 lambdas (more if you write to support). Lambda works in 11 regions. So you can run in parallel more than 2000 lambdas, each of which will conduct load testing of your service. Five minutes of such testing will cost just one dollar.

#### Demo

Demo in this package tries to send requests to github.com for 5 seconds with 1 connection and also conduct pyresttest dummy test.

#### Tools

1. WRK (https://github.com/wg/wrk) - the main tool for load testing. It works with multiple threads, you can specify the number of connections and length of the load. For more fine-tuning, you can use LuaJIT scripts (https://www.lua.org/).
2. Pyrestest (https://github.com/svanoort/pyresttest) is a handy tool for testing the full pipeline of the API. For example, the user registers, then uses the api key to create tasks / make notes / downloads files, then reads them, then deletes them.

#### Documentation

https://github.com/wg/wrk

https://github.com/svanoort/pyresttest

---
### Lxml + requests

#### Intro

Package for parsing static HTML pages. Difference here is that it works faster and consumes less memory than PhantomJS but is limited in terms websites it can parse and other features.

#### Serverless start

```
serverless install -u https://github.com/ryfeus/lambda-packs/tree/master/Lxml_requests/source -n lxml-requests
cd lxml-requests
serverless deploy
serverless invoke --function main --log
```

#### Build pack

```
wget https://github.com/ryfeus/lambda-packs/blob/master/Lxml_requests/buildPack.sh
docker pull amazonlinux:latest
docker run -v $(pwd):/outputs --name lambdapackgen -d amazonlinux:latest tail -f /dev/null
docker exec -i -t lambdapackgen /bin/bash /outputs/buildPack.sh
```

#### Tools

Lxml 3.7.1

#### Documentation

http://lxml.de/

---
### Tensorflow

#### Intro 

Open source library for Machine Intelligence. Basically revolutionized AI and made it more accessible. Using tensorflow on lambda is not as bad as it may sound - for some simple models it is the simplest and the cheapest way to deploy.

#### Demo

As hello world code I used recognition of images trained on imagenet (https://www.tensorflow.org/tutorials/image_recognition). Given the price tag lambda one run (recognition of one picture) will cost $0.00005. Therefore for a dollar you can recognize 20,000 images. It is much cheaper than almost any alternatives, though completely scalable (200 functions can be run in parallel), and can be easily integrated into cloud infrastructure. Current demo downloads image from link 'imagelink' from event source ( if empty - then downloads https://s3.amazonaws.com/ryfeuslambda/tensorflow/imagenet/cropped_panda.jpg)

#### Tools

Tensorflow 1.4.0

#### Documentation

https://www.tensorflow.org/tutorials/image_recognition

#### Nightly version

Nightly version archive is more than 50 MB in size but it is still eligible for using with AWS Lambda (though you need to upload pack through S3). For more read here:

https://hackernoon.com/exploring-the-aws-lambda-deployment-limits-9a8384b0bec3

#### Serverless start

```
serverless install -u https://github.com/ryfeus/lambda-packs/tree/master/tensorflow/source -n tensorflow
cd tensorflow
serverless deploy
serverless invoke --function main --log
```

#### Build pack

for Python2:

```bash
wget https://raw.githubusercontent.com/ryfeus/lambda-packs/master/Tensorflow/buildPack.sh
wget https://raw.githubusercontent.com/ryfeus/lambda-packs/master/Tensorflow/index.py
docker pull amazonlinux:latest
docker run -v $(pwd):/outputs --name lambdapackgen -d amazonlinux:latest tail -f /dev/null
docker exec -i -t lambdapackgen /bin/bash /outputs/buildPack.sh
```

for Python3:

```bash
wget https://raw.githubusercontent.com/ryfeus/lambda-packs/master/Tensorflow/buildPack_py3.sh
wget https://raw.githubusercontent.com/ryfeus/lambda-packs/master/Tensorflow/index_py3.py
docker pull amazonlinux:latest
docker run -v $(pwd):/outputs --name lambdapackgen -d amazonlinux:latest tail -f /dev/null
docker exec -i -t lambdapackgen /bin/bash /outputs/buildPack_py3.sh
```

> Note: Remember You should set `python3.6` for AWS Lambda function environment.

---
### Sklearn

#### Intro

Package for fans of machine learning, building models and the like. I doubt that there is a more convenient way to deploy model to the real world.

#### Tools

1. Scikit-learn 0.17.1
2. Scipy 0.17.0

#### Documentation

http://scikit-learn.org/

---
### Skimage

#### Intro

Package of image processing tools, and not only to style image, but also a large set of computer vision algorithms.

There are currently two zipped packs available, Pack.zip and Pack_nomatplotlib.zip, you probably want to use Pack_nomatplotlib.zip. See https://github.com/ryfeus/lambda-packs/issues/5 for more information.

#### Tools

Scikit-image 0.12.3

#### Documentation

http://scikit-image.org/

---
### OpenCV + PIL

#### Intro

Another package of image processing tools, and not only to style image, but also a large set of Computer vision algorithms.

#### Tools

1. OpenCV 3.1.0
2. PIL 4.0.0

#### Documentation

https://pillow.readthedocs.io/

http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html

---
### Pandas

#### Intro

Package for fans of statistics, data scientists and data engineers. RAM at lambda is 1.5 gigabytes, and the maximum operating time - 5 minutes. I am sure that will be enough for most tasks.

#### Tools

Pandas 0.19.0

#### Documentation

http://pandas.pydata.org/

---
### Spacy

#### Intro

Opensource library for Natural Language Processing in python.

#### Tools

1. Spacy 2.0.11

#### Documentation

https://spacy.io/

#### Example

Example code loads language model from S3 and uses it to analyze sentence.

---
### Tesseract

#### Intro

OCR (optical character recognition) library for text recognition from the image.

#### Documentation

https://github.com/tesseract-ocr/tesseract

---

### PDF generator + Microsoft office file generator (docx, xlsx, pptx) + image generator (jpg, png) + book generator (epub)

#### Intro

"Hello world" code in package creates example of every document. Basically these libs are low memory (less than 128MB) and high speed (less than 0.5 seconds) so it's something like ~1m documents generated per 1$ in terms of AWS Lambda pricing.

#### Tools

- docx (python-docx - https://pypi.python.org/pypi/python-docx)
- xlsx (XlsxWriter - https://pypi.python.org/pypi/XlsxWriter)
- pptx (python-pptx - https://pypi.python.org/pypi/python-pptx)
- pdf (Reportlab - https://pypi.python.org/pypi/reportlab)
- epub (EbookLib - https://pypi.python.org/pypi/EbookLib)
- png/jpg/... (Pillow - https://pypi.python.org/pypi/Pillow)


---

### Satellite imagery processing (rasterio + OSGEO + pyproj + shapely + PIL)

#### Intro

AWS Lambda pack in Python for processing satellite imagery. Basically it enables to deploy python code in an easy and cheap way for processing satellite imagery or polygons. In “hello world” code of the pack I download red, green, blue Landsat 8 bands from AWS, make True Color image out of it and upload it to S3. It takes 35 seconds and 824MB of RAM for it so ~2500 scenes can be processed for 1$.

#### Tools

- Rasterio (https://github.com/mapbox/rasterio 0.36)
- OSGEO (https://trac.osgeo.org/gdal/wiki/GdalOgrInPython)
- Pyproj (https://github.com/jswhit/pyproj)
- Shapely (https://github.com/Toblerity/Shapely)
- PIL (https://pillow.readthedocs.io/)
