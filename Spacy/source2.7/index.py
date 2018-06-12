import spacy
import boto3
import os


def download_dir(client, resource, dist, local='/tmp', bucket='s3bucket'):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                if not os.path.exists(os.path.dirname(local + os.sep + file.get('Key'))):
                     os.makedirs(os.path.dirname(local + os.sep + file.get('Key')))
                resource.meta.client.download_file(bucket, file.get('Key'), local + os.sep + file.get('Key'))

def handler(event, context):
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    # download_dir(client, resource, 'en-sm', '/tmp','ryfeus-spacy')
    if (os.path.isdir("/tmp/en_core_web_sm")==False):
        download_dir(client, resource, 'en_core_web_sm', '/tmp','ryfeus-spacy')
    print(os.listdir('/tmp'))
    print(os.listdir('/tmp/en_core_web_sm/en_core_web_sm-2.0.0'))
    spacy.util.set_data_path('/tmp')
    nlp = spacy.load('/tmp/en_core_web_sm/en_core_web_sm-2.0.0')
    doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
    for token in doc:
        print(token.text, token.pos_, token.dep_)
    return 'finished'