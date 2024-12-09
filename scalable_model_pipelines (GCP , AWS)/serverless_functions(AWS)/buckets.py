import boto3
import os

def credentials(region=None):
    if region != None:
        region = region

    import json
    creds = json.load(open('credentials.json','rb'))
    aws_access_key_id = creds['aws_access_key_id']
    aws_secret_access_key = creds['aws_secret_access_key']

    location = {'LocationConstraint': region}
    session = boto3.Session(
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        region_name=region
    )

    client_aws = session.client('s3',
                              aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,region_name=region)

    return client_aws , location

def create_bucket(client , bucket_name,loc_constraint):

    client.create_bucket(Bucket=bucket_name,
                             CreateBucketConfiguration=loc_constraint)
    return True

def upload_data(client,bucket_name,file_path,key_name='default'):

    client.upload_file(file_path,bucket_name,key_name)

    return True


if __name__ == "__main__":
    client , location = credentials(region="us-west-1")
    file_path = os.path.join(os.getcwd(),'lamda.zip')
    upload_data(client,'educative-2',file_path,key_name='lamda_1.zip')
