#!/usr/bin/env python
import httplib2
import datetime
import time
import os
import selenium 
import json
import boto3
import requests
from dateutil.parser import parse
from selenium import webdriver 
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from apiclient.discovery import build
from oauth2client.client import GoogleCredentials

def handler(event, context):
	# set user agent
	user_agent = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36")

	dcap = dict(DesiredCapabilities.PHANTOMJS)
	dcap["phantomjs.page.settings.userAgent"] = user_agent
	dcap["phantomjs.page.settings.javascriptEnabled"] = True

	browser = webdriver.PhantomJS(service_log_path=os.path.devnull, executable_path="/var/task/phantomjs", service_args=['--ignore-ssl-errors=true'], desired_capabilities=dcap)
	browser.get('https://en.wikipedia.org/wiki/Special:Random')
	line = browser.find_element_by_class_name('firstHeading').text
	print(line)
	return line