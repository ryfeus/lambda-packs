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

### Selenium_PhantomJS

Useful for web testing and scraping.

#### Demo

Current demo opens random page from wiki (https://en.wikipedia.org/wiki/Special:Random) and prints title