import boto3
import boto3.session
from .logger import logger


class S3Client:

    def __init__(self, **kwargs):
        self.s3_resource = boto3.resource('s3',
                                          aws_access_key_id=kwargs['access_key'],
                                          aws_secret_access_key=kwargs['secret_key'],
                                          endpoint_url=kwargs['endpoint'])

        self.my_bucket = self.s3_resource.Bucket(kwargs['bucket'])
        self.bucket_name = kwargs['bucket']

    def walk(self, prefix: str, call_back, **kwargs):
        self.__walk_s3(prefix, call_back, **kwargs)

    def download(self, s3_file, save_as):
        self.my_bucket.download_file(s3_file, save_as)

    def upload(self, file_path_to_upload, save_as):
        self.my_bucket.upload_file(file_path_to_upload, save_as)

    def __walk_s3(self, prefix: str, call_back, **kwargs):
        objects = self.my_bucket.objects.filter(Prefix=prefix)
        for obj in objects:
            call_back(obj.key, **kwargs)

    @staticmethod
    def s3_mule(**kwargs):
        s3 = S3Client(endpoint=kwargs['endpoint'], secret_key=kwargs['secret_key'],
                      access_key=kwargs['access_key'], bucket=kwargs['bucket'])
        mule = {'download': s3.download, 'upload': s3.upload}

        if kwargs['files'] is None:
            exit(0)

        if kwargs['action'] not in mule:
            exit(1)

        for rule in kwargs['files'].split(","):
            rule_split = rule.split(':')
            logger.info(f"{kwargs['action']} start of: {rule_split[0]}, {rule_split[1]}...")
            try:
                mule[kwargs['action']](rule_split[0], rule_split[1])
            except Exception as e:
                logger.info(f"{kwargs['action']} failed")
            finally:
                logger.info('done')


if __name__ == '__main__':
    S3Client.s3_mule()


# https://boto3.amazonaws.com/v1/documentation/api/1.16.53/guide/s3-uploading-files.html
# https://stackoverflow.com/questions/50105094/python-upload-large-files-s3-fast
# https://cloudplexo.com/uploading-large-files-upto-5tb-to-amazon-s3-using-boto3-in-python