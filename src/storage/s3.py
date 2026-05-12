import boto3
from botocore.client import Config
from pathlib import Path
import logging

class MinioStorage:
    def __init__(self, config):
        s3_config = config.get("minio", {})
        self.endpoint = s3_config.get("endpoint", "localhost:9000")
        self.access_key = s3_config.get("access_key", "minioadmin")
        self.secret_key = s3_config.get("secret_key", "minioadmin")
        self.bucket_name = s3_config.get("bucket_name", "tree-images")
        self.secure = s3_config.get("secure", False)
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=f"{'https' if self.secure else 'http'}://{self.endpoint}",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # Minio mặc định
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except:
            self.s3.create_bucket(Bucket=self.bucket_name)
            print(f"Created bucket: {self.bucket_name}")

    def upload_file(self, local_path: Path, object_name: str = None):
        if object_name is None:
            object_name = local_path.name
            
        try:
            self.s3.upload_file(str(local_path), self.bucket_name, object_name)
            return object_name
        except Exception as e:
            logging.error(f"Error uploading to Minio: {e}")
            return None

    def get_url(self, object_name: str, expires_in: int = 3600):
        """Tạo Presigned URL để hiển thị ảnh trên Web"""
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logging.error(f"Error generating URL: {e}")
            return None
