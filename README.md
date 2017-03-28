# lambda-packs 

Precompiled packages for AWS lambda

## How to start

1. https://aws.amazon.com/lambda/ and create/sign in into account
2. Lambda > Functions - Create lambda function
3. Black function
4. Configure triggers - Next
5. Configure function
  - Runtime - Python 2.7
6. Lambda function handler and role
  - Handler - service.lambda_handler
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

Tensorflow 1.0.0

#### Documentation

https://www.tensorflow.org/tutorials/image_recognition

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
### Tesseract
