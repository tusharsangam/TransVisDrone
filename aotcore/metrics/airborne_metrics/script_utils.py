# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import sys

def setup_logging(log_level):
    """Utility function setting up the logger"""
    logging.basicConfig(format='%(asctime)-15s:%(levelname)s:%(filename)s:%(lineno)d %(message)s',
                        level=logging.getLevelName(log_level),
                        stream=sys.stderr)

def assert_file_format (filename, allowed_format=None):
    """This function asserts that provided filename end with allowed format, if None is provided
    default formats are: .csv, .json, .json.gz
    """
    if allowed_format is None:
        assert filename.endswith('.csv') or filename.endswith('.json') or filename.endswith('.json.gz'), (
            'Unsupported file format, please provide .csv (preferred) or .json or .gz')
    else:
        assert filename.endswith(allowed_format), '{} format expected'.format(allowed_format)

def remove_extension(filename):
    return filename.replace('.json', '').replace('.csv', '').replace('.gz', '')
