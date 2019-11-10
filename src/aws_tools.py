import logging
import boto3
from botocore.exceptions import ClientError


def upload_to_s3(filename, bucket, channel, object_name=None):
    """
    Stores the file in the s3 bucket and 'channel' (folder) specified
    If specified, object_name will the be name in S3
    """
    subfolder, buc_folders = None, None
    if object_name is None:
        object_name = filename
    if "/" in bucket:
        buc_folders = bucket.split("/")
    elif "\\" in bucket:
        buc_folders = bucket.split("\\")
    if buc_folders:
        bucket = buc_folders[0]
        subfolder = buc_folders[1:]
    if isinstance(bucket, list):
        bucket, subfolder = bucket[0], bucket[1:]

    if subfolder:
        key = "/".join(subfolder) + "/" + channel + "/" + object_name
    else:
        key = channel + "/" + object_name

    # Read file
    with open(filename, "rb") as data:
        s3 = boto3.resource("s3")
        s3.Bucket(bucket).put_object(Key=key, Body=data)


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def read_from_s3(bucketname, itemname):
    s3 = boto3.resource("s3")
    obj = s3.Object(bucketname, itemname)
    return obj.get()['Body'].read()


def download_from_s3(bucketname, filename, local_file):
    """Download the filename to a local_file"""
    s3 = boto3.resource("s3")
    s3.Bucket(bucketname).download_file(filename, local_file)
