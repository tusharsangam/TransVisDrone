# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import gzip
import json
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def is_in_df_columns(df, col_names):
    """check if columns with col_names exist in data_frame df"""
    all_cols = df.columns.to_list()
    return np.all([col_name in all_cols for col_name in col_names])

def _fix_empty_records_json_dict(json_data, record_key):
    """adds empty dictionary for each empty list in record_key 
    of each element of json_data"""
    for element in json_data:
        if not(element[record_key]):
            element[record_key] = [{}]
    return json_data

def normalize_json_deeplearning_groundtruth_to_dataframe(json_data):
    """Custom function that normalizes json into dataframe"""
    # it is important to keep all the images in the data frame to know how many 
    # images were evaluated, hence we fix such images to have 'gt'=[{}]
    df = pd.json_normalize(json_data['samples'].values(), record_path='entities', meta=[
                ['metadata', 'resolution', 'width'] , ['metadata', 'resolution', 'height']],
                                                         sep='_', errors='ignore')
    df = df.assign(gt_left = [bb[0] if bb==bb else bb for bb in df['bb']])
    df = df.assign(gt_top = [bb[1] if bb==bb else bb for bb in df['bb']])
    df = df.assign(gt_right = [bb[0] + bb[2] if bb==bb else bb for bb in df['bb']])
    df = df.assign(gt_bottom = [bb[1] + bb[3] if bb==bb else bb for bb in df['bb']])
    df = df.drop(columns=['bb'])
    df = df.rename(columns={'labels_is_above_horizon': 'is_above_horizon', 
        'blob_range_distance_m': 'range_distance_m', 'blob_frame': 'frame', 
        'metadata_resolution_width': 'size_width', 'metadata_resolution_height': 'size_height',
        })
    return df

def normalize_json_result_to_dataframe(json_data):
    """Custom function that normalizes json into dataframe"""
    if len(json_data) == 0:
        return None
    df = pd.json_normalize(json_data, record_path='detections',
                            meta=['img_name'],
                            sep='_', errors='ignore')
    df_columns = df.columns.to_list()
    columns_to_return = ['img_name', 'n', 'x', 'y', 'w', 'h', 's']
    if 'track_id' in df_columns:
        columns_to_return.append('track_id')
    if 'object_id' in df_columns:
        columns_to_return.append('object_id')
    return df[columns_to_return]

def _get_as_dataframe(filename, normalization_func=None):
    """Reads the provided .json/.csv filename
    if needed normalizes json into csv using the provided normalization_func
    returns data frame representation
    """  
    log.info('Reading provided %s', filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.json') or filename.endswith('.json.gz'): 
        if normalization_func is None:
            raise ValueError('Please provide normalization function for you json schema')
        if filename.endswith('.json'):
            log.info('Loading .json')
            with open(filename, 'r') as json_data:
                json_gt = json.load(json_data)
        else:
            log.info('Loading .json.gz')
            with gzip.open(filename, 'rt', encoding='UTF-8') as json_data:
                json_gt = json.load(json_data)
        log.info('Normalizing json. This operation is time consuming. The result .csv will be saved ' 
                'Please consider providing .csv file next time')
        df = normalization_func(json_gt)
    else:
        raise ValueError('Only .csv, .json or .json.gz are supported')
    return df   

def get_deeplearning_groundtruth_as_data_frame(deeplearning_groundtruth_filename):
    """Reads the deep learning ground truth as provided .json/.csv
    if needed normalizes json into csv
    returns data frame representation of the deep learning ground truth
    """  
    log.info('Reading ground truth')
    return _get_as_dataframe(deeplearning_groundtruth_filename, 
                           normalize_json_deeplearning_groundtruth_to_dataframe)

def get_results_as_data_frame(detection_results_filename):
    """Reads detection results as provided .json/.csv
    if needed normalizes json into csv
    returns data frame representation of the detection results
    """  
    log.info('Reading detection results')
    return _get_as_dataframe(detection_results_filename, 
                             normalize_json_result_to_dataframe)
    