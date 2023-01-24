import os
from urllib.parse import urlparse
import cv2
from PIL import Image
import numpy
import multiprocessing
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from loguru import logger


class FileHandler:
    def __init__(self, local_path, s3_path, download_if_required=True):
        self.local_path = local_path
        self.download_if_required = download_if_required
        self.s3_client = self.init_s3_client()
        self.s3_bucket, self.s3_path = self.parse_s3_path(s3_path)

    @staticmethod
    def init_s3_client():
        s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        return s3_client

    @staticmethod
    def parse_s3_path(s3_path):
        o = urlparse(s3_path, allow_fragments=False)
        return o.netloc, o.path.lstrip('/')

    def download_file_if_needed(self, s3_path, local_path):
        if not self.file_exist_locally(local_path):
            if self.download_if_required:
                logger.info("[download_from_s3] File not found locally, downloading: %s" % s3_path)
                self.download_from_s3(s3_path, local_path)

        return self.file_exist_locally(local_path)

    def get_file_content(self, s3_path, type='None', local_path=None):
        if not local_path:
            local_path = s3_path

        if not self.download_file_if_needed(s3_path, local_path):
            raise FileNotFoundError

        full_path = self.absolute_path_to_file_locally(local_path)
        if type.lower() == "cv2":
            return cv2.imread(full_path)

        if type.lower() == "pil":
            return Image.open(full_path)

        if type.lower() == "numpy":
            return numpy.asarray(Image.open(full_path))

        return open(full_path).read()

    def create_local_directory(self, path):
        if type(path) == list:
            path = path[1]
        os.makedirs(os.path.dirname(self.absolute_path_to_file_locally(path)), exist_ok=True)

    def absolute_path_to_file_locally(self, path):
        return os.path.join(self.local_path, path)

    def absolute_path_to_file_on_s3(self, path):
        return os.path.join(self.s3_path, path)

    def file_exist_locally(self, path):
        return os.path.isfile(self.absolute_path_to_file_locally(path))

    def download_from_s3(self, s3_path, local_path):
        self.create_local_directory(local_path)
        self._download_from_s3([self.s3_bucket, self.absolute_path_to_file_on_s3(s3_path),
                               self.absolute_path_to_file_locally(local_path)])

    @staticmethod
    def _download_from_s3(config):
        if os.path.isfile(config[2]):
            return
        s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        s3_client.download_file(config[0], config[1], config[2])

    def download_from_s3_parallel(self, paths, parallel=None):
        if parallel is None:
            parallel = multiprocessing.cpu_count()

        self.create_local_directory(paths[0])

        prepare = []
        for path in paths:
            if type(path) == list:
                prepare.append([self.s3_bucket, self.absolute_path_to_file_on_s3(path[0]),
                                self.absolute_path_to_file_locally(path[1])])
            else:
                prepare.append([self.s3_bucket, self.absolute_path_to_file_on_s3(path),
                                self.absolute_path_to_file_locally(path)])
        with multiprocessing.Pool(parallel) as pool:
            pool.map(self._download_from_s3, prepare)


if __name__ == "__main__":
    local_path = '/Users/skbly7/Terminal/aicrowd/repos/airborne-detection-starter-kit/data'
    s3_path_ = 's3://airborne-obj-detection-challenge-training/part1/'
    file_handler = FileHandler(local_path, s3_path_)
