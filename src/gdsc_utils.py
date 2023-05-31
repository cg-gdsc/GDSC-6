"""
This is the utils module.

This module contains functions that help downloading an uploading the data from and to s3 buckets, as well as creating a bucket.
"""

import boto3
import logging
import os
import tarfile
from botocore.exceptions import ClientError
from pathlib import Path
from tqdm import tqdm

from config import DEFAULT_BUCKET, DEFAULT_REGION

PROJECT_DIR = Path(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))


def upload_to_s3(local_path: str, 
                 s3_path: str, 
                 bucket: str) -> str:
    """
    Upload a file from a local directory to an S3 bucket.

    Args:
        local_path (str): The path of the local file to upload.
        s3_path (str): The S3 path to upload the file to, relative to the bucket name.
        bucket (str): The name of the S3 bucket.

    Returns:
        str: The remote path to the uploaded file in the S3 bucket.

    Raises:
        None
    """
    client = boto3.client("s3")
    client.upload_file(local_path, bucket, s3_path)
    return f"s3://{bucket}/{s3_path}"


def download_file(s3_path: str, 
                  local_dir: str, 
                  bucket: str = DEFAULT_BUCKET, 
                  region: str = DEFAULT_REGION)  -> str:
    """
    Download a file from an S3 bucket to a local directory.

    Args:
        s3_path (str): The S3 path to the file, relative to the bucket name.
        local_dir (str): The local path to save the downloaded file. If set to None it will download the file to the root directory
        bucket (str): The name of the S3 bucket (default: DEFAULT_BUCKET).
        region (str): The region in which the bucket is located (default: DEFAULT_REGION).

    Returns:
        str: The local file path where the file is downloaded.
    """
    logger = logging.getLogger(__name__)
    s3 = boto3.resource("s3", region_name=region)
    file_name = s3_path.split("/")[-1]
    f = f"{PROJECT_DIR}/{local_dir}/{file_name}" if local_dir else f"{PROJECT_DIR}/{file_name}"
    if os.path.isfile(f):
        logger.info(f"File {s3_path} already exists. Skipping download")
        return f
    s3.Bucket(bucket).download_file(s3_path, f)
    return f


def download_directory(s3_path: str, 
                       local_dir: str, 
                       bucket: str = DEFAULT_BUCKET, 
                       region: str = DEFAULT_REGION) -> None:
    """
    Download a folder from an S3 bucket to a local directory.

    Args:
        s3_path (str): The S3 path to the folder, relative to the bucket name.
        local_dir (str): The local path to save the downloaded files. If set to None it will download the files to the root directory
        bucket (str): The name of the S3 bucket (default: DEFAULT_BUCKET).
        region (str): The location/region of the S3 bucket (default: DEFAULT_REGION).

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    s3_resource = boto3.resource("s3", region_name=region)
    bucket = s3_resource.Bucket(bucket)
    local_abs_path = PROJECT_DIR / local_dir if local_dir else PROJECT_DIR

    logger.info(f"Downloading {s3_path} to {local_abs_path}")
    for obj in tqdm(bucket.objects.filter(Prefix=s3_path)):
        f = f"{local_abs_path}/{obj.key}"
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        if obj.key == s3_path or os.path.isfile(f):
            continue
        bucket.download_file(obj.key, f)
    logger.info(f"Downloaded {s3_path} to {local_abs_path}")


def create_encrypted_bucket(bucket_name: str) -> None:
    """
    Creates an encrypted S3 bucket with the specified bucket name.

    Args:
        bucket_name (str): The name of the S3 bucket to be created.

    Returns:
        None
    """
    s3 = boto3.client("s3")
    try:
        s3.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"].get("Code") not in (
            "BucketAlreadyExists",
            "BucketAlreadyOwnedByYou",
        ):
            raise e

    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )


def download_and_extract_model(model_uri: str, 
                               local_dir: str) -> str:
    """
    Downloads a model tarfile from the specified model_uri (S3) and extracts it to the local directory.

    Args:
        model_uri (str): The URI of the model tarfile (S3 path).
        local_dir (str): The local directory where the model will be extracted.

    Returns:
        str: The local directory path where the model is extracted.
    """
    bucket, key = model_uri.replace("s3://", "").split("/", 1)
    exp_name = key.split("/")[1]
    local_model_dir = f"{local_dir}/{exp_name}"
    os.makedirs(local_model_dir, exist_ok=True)
    file_path = download_file(s3_path=key, bucket=bucket, local_dir=local_model_dir)
    with tarfile.open(file_path, mode="r") as tar:
        tar.extractall(local_model_dir)
    return local_model_dir
