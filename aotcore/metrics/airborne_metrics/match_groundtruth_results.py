# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
""" This module performs matching between detection results and ground truth
using pandas API outer merge between data frame of ground truth and detection 
results.
All the ground truth with the same image and object names are merged to all the 
detection results with the same image and object names, creating all the possible 
combinations, based on extended IoU, which  comes to alleviate the extra sensitivity of small objects 
to IOU, by extending all the small objects and evaluated detection results to 
have a minimum area specified in pixels (while maintaining the original aspect ratio)
For comparison, original iou is calculated and saved as well

INPUT:
ground truth
results 
NOTE: the matching will not be performed if there are images in results that
do not appear in the ground truth

OUTPUT:
Data frame - outer join of groundtruth and detection results with 
match/no match between those based 
on the chosen matching function and thresholds  
"""
import argparse
from functools import partial
import json
import logging
import numpy as np
import os
import pandas as pd

from .pandas_utils import is_in_df_columns, get_deeplearning_groundtruth_as_data_frame, get_results_as_data_frame
from .script_utils import setup_logging, assert_file_format

#############################################
# Defaults
##############################################
TWO_LINES_TOL_ = 10 # pixels
IS_MATCH_MAX_DISTANCE_ = 10
LARGE_DIST_ = np.iinfo(np.int32(10)).max
EPSILON_ = 1e-6
MIN_OBJECT_AREA_ = 100 # pixels
DEFAULT_IOU_IS_MATCH_ = 0.2
DEFAULT_IOU_IS_NO_MATCH_ = 0.02
MAX_FRAMES_PER_FLIGHT = 1200
MIN_TRACK_LEN_ = 0

# Columns 
DETECTION_BBOX_COLS_ = ['det_right', 'det_left', 'det_bottom', 'det_top']  
GROUNDTRUTH_BBOX_COLS_ = ['gt_right', 'gt_left', 'gt_bottom', 'gt_top'] 
RESULTS_NAME_PREFIX_ = 'gt_det'
##############################################
# Script related code
##############################################
log = logging.getLogger(__name__)

def add_flags(parser):
    """Utility function adding command line arguments to the parser"""
    # input files 
    parser.add_argument('--deeplearning-groundtruth', '-g', required=True, 
                        help='Path to the ground truth .json or .csv file' 
                        ' (consider providing the groundtruth in .csv format)')
    parser.add_argument('--airborne-classifier-results', '-r', required=True, 
                        help='Path to the detection results .json or .csv file')
    # output  
    parser.add_argument('--output-dir-path', '-o', 
                        help='Desired folder to save the output data frame with '
                        'match/no match between groundtruth and detections')
    parser.add_argument('--results-name-prefix', type=str, default=RESULTS_NAME_PREFIX_,
                        help='Prefix for results filename')
    # matching algorithm and its parameters
    parser.add_argument('--extend-small-detections', '-e', action='store_true', 
                        help='Specify if the airborne classifier detection results '
                        'should be extended to minimum area')
    parser.add_argument('--minimum-object-area', '-a', default=MIN_OBJECT_AREA_, type=int,
                        help='Minimum object area, specify if need to extend'
                        'ground truth and detections to have this minimum area')
    parser.add_argument('--is-match-threshold', '-p', type=float, default=DEFAULT_IOU_IS_MATCH_,
                        help='Threshold for ground truth and detection '
                        'to be considered a "match"')
    parser.add_argument('--is-no-match-threshold', '-n', type=float, default=DEFAULT_IOU_IS_NO_MATCH_,
                        help='Threshold for ground truth and detection '
                        'to be considered a "no match"')
    # detection filtering
    parser.add_argument('--detection-score-threshold', '-t', type=float, default=0.0,
                        help='Threshold for filtering detections before matching')
    parser.add_argument('--min-track-len', type=int, default=MIN_TRACK_LEN_,
                         help='Minimum length of track to include in results')

def _assert_non_negative_threshold(threshold):
    """assertion helper"""
    assert threshold >= 0.0, 'Threshold for matching algorithm, is expected to be non-negative'

def check_flags(flags):
    """Utility function to check the input"""
    assert_file_format(flags.deeplearning_groundtruth)
    assert_file_format(flags.airborne_classifier_results)
    _assert_non_negative_threshold(flags.is_match_threshold)
    if flags.is_no_match_threshold is not None: 
        _assert_non_negative_threshold(flags.is_no_match_threshold)
    assert flags.output_dir_path is None or not os.path.isfile(flags.output_dir_path),(
                                                        'Directory name is expected as output path')

##################################################
# Groundtruth - detections matching code
##################################################
def _limit_bbox_to_image_size(df, prefix):
    # """helper function to bring bounding box values within image size limits)"""
    df['{}_top'.format(prefix)].clip(0, df['size_height'] - 1, inplace=True)
    df['{}_bottom'.format(prefix)].clip(0, df['size_height'] - 1, inplace=True)
    df['{}_left'.format(prefix)].clip(0, df['size_width'] - 1, inplace=True)
    df['{}_right'.format(prefix)].clip(0, df['size_width'] - 1, inplace=True)
    return df

def augment_with_detection_top_bottom_left_right(df):
    """ Add 4 columns to the data frame that correspond to 
    top (y1), bottom(y2), left(x1), right(x2) of the detection 
    bounding box
    """
    required_columns = ['x', 'y', 'h', 'w']
    required_columns_str = ', '.join(required_columns) 
    assert is_in_df_columns(df, required_columns), (
        'One or more out of {} columns is not found, '
        'cannot perform augmentation with bounding box'.format(required_columns_str))
    half_height = df['h'] / 2
    half_width = df['w'] / 2
    df = df.assign(det_top = df['y'] - half_height)
    df = df.assign(det_bottom = df['y'] + half_height)
    df = df.assign(det_left = df['x'] - half_width)
    df = df.assign(det_right = df['x'] + half_width)
    return df

def _calc_bbox_area(df, prefix):
    return ((df['{}_right'.format(prefix)] - df['{}_left'.format(prefix)]) 
           * (df['{}_bottom'.format(prefix)] - df['{}_top'.format(prefix)]))
      
def augment_with_detection_area(df):
    """Augments data frame with detection area"""
    required_columns_str = ', '.join(DETECTION_BBOX_COLS_) 
    assert is_in_df_columns(df, DETECTION_BBOX_COLS_), (
        'One or more out of {} columns is not found, '
                        'cannot calculate area'.format(required_columns_str))
    df = df.assign(det_area = _calc_bbox_area(df, 'det'))
    return df

def augment_with_groundtruth_area(df):
    """Augments data frame with ground truth area"""
    required_columns_str = ', '.join(GROUNDTRUTH_BBOX_COLS_) 
    assert is_in_df_columns(df, GROUNDTRUTH_BBOX_COLS_), (
        'One or more out of {} columns is not found, '
                        'cannot calculate area'.format(required_columns_str))

    df = df.assign(gt_area = _calc_bbox_area(df, 'gt'))
    return df

def augment_with_iou(df):
    """Augments data frame with iou between the detection and groundtruth"""
    required_columns = GROUNDTRUTH_BBOX_COLS_ + DETECTION_BBOX_COLS_ + ['det_area', 'gt_area'] 
    required_columns_str = ', '.join(required_columns) 
    assert is_in_df_columns(df, required_columns), (
        'One or more out of {} columns is not found'
                     'cannot perform thresholding'.format(required_columns_str))
    df['iou'] = 0
    ix_min = df[['det_left', 'gt_left']].max(axis=1)
    iy_min = df[['det_top', 'gt_top']].max(axis=1)
    ix_max = df[['det_right', 'gt_right']].min(axis=1)
    iy_max = df[['det_bottom', 'gt_bottom']].min(axis=1)

    iw = np.maximum(ix_max - ix_min, 0.)
    ih = np.maximum(iy_max - iy_min, 0.)
 
    intersections = iw * ih
    unions = (df['det_area'] + df['gt_area'] - intersections)

    ious = intersections / unions
    ious[unions < 1e-12] = 0
    # the iou is set to zero for frame where there is no ground truth (ground truth area is NaN )
    # and there is a detection (detection are is not NaN)
    # if there is not detection in a frame the iou will be NaN
    ious[df['gt_area'].isnull() & df['det_area'].notnull()] = 0
    df = df.assign(iou = ious)
    return df

def _augment_with_match_no_match(df, is_match, is_no_match):
    assert np.all((is_match & is_no_match) == False), (
        'the same combination of ground truth and detection cannot be both match and no match')
    df['gt_det_match'] = 0
    df['gt_det_no_match'] = 0
    df.loc[is_match, 'gt_det_match'] = 1 
    df.loc[is_no_match, 'gt_det_no_match'] = 1
    return df 

def augment_with_iou_match(df, is_match_min_iou, is_no_match_max_iou=None):
    """Augments the data frame with match/ no match based on the iou"""
    assert is_in_df_columns(df, ['iou', 'Exists']), (
        'One or more out of iou, Exists columns is not found, ' 
                                'cannot perform assignment of match/ no match')
    if is_no_match_max_iou is None:
        is_no_match_max_iou = is_match_min_iou
    log.info('IoU matching: match minimum iou = %.2f, and no match '
            'maximum iou = %.2f ', is_match_min_iou, is_no_match_max_iou)
    
    # the detection and ground truth are matched if the iou is above a certain threshold
    is_match = df['iou'] >= is_match_min_iou
    # the detection and ground truth are NOT matched, if there is a detection 
    # (the condition df['Exists'] != 'left_only' checks that there is a detection in a frame) 
    # and its IOU to ground truth is below a certain threshold
    is_no_match = (df['Exists'] != 'left_only') & (df['iou'] < is_no_match_max_iou)
    df = _augment_with_match_no_match(df, is_match, is_no_match)
    return df
   
def augment_with_detection_id(df_results):
    """Enumerates the data frame of results with detection id""" 
    df_results = df_results.assign(detection_id = list(range(len(df_results))))
    return df_results

def _extend_bounding_boxes(orig_box_width, orig_box_height, min_box_area):
    """Helper function: extends small bounding boxes to have the specified minimum object area, 
    while maintaining original aspect ratio.
    Note this function assumes all the provided boxes have area less than minimun area
    Formula: 
    1) new_area = width * height
    2) aspect_ratio = width / height (also aspect_ratio = orig_width / orig_height)
    ==> height  = width / aspect_ratio
    ==> new_area = width * (width / aspect_ratio)
    ==> sqrt(new_area * aspect_ratio) = width 
    Params:
        Original bounding box widths and heights and minimum bounding box area to get after extension
    Throws:
        ValueError if any of the provided bounding boxes has greater area than minimum box area
        or has widths or heights equal to zero
    Returns: 
        Extended width sand heights and the corresponding deltas with respect to the original widths
        and heights
    """ 
    if not np.all(orig_box_width * orig_box_height < min_box_area):
        raise ValueError('This function expects all the original areas to be '
                                                    'less then the minimum area')
    if not np.all(orig_box_width > 0):
        raise ValueError('This function expects non-zero width of bounding boxes')
    if not np.all(orig_box_height > 0):
        raise ValueError('This function expects non-zero height of bounding boxes')
    orig_aspect_ratio = (orig_box_width / orig_box_height).astype('float')
    extended_width = np.sqrt(min_box_area * orig_aspect_ratio)
    extended_height = min_box_area / extended_width
    delta_width = extended_width - orig_box_width
    delta_height = extended_height - orig_box_height    
    assert np.all(delta_width >= 0), 'extention should yield bigger or equal width'
    assert np.all(delta_height >= 0), 'extention should yield bigger or equal height'                                    
    return delta_width, delta_height, extended_width, extended_height

def _extend_bounding_boxes_to_have_minimum_object_area(df, prefix, 
                                            bboxs_to_extend, min_object_area):
    """Helper function: extends specified bounding boxes of a data frame to have
    minimum object area
    The specification is done based on provided parameters
        bboxs_to_extend: indexes in data frame 
        prefix: 'gt' for ground truth and 'det' for detection
    """
    bbox_width = df['{}_right'.format(prefix)] - df['{}_left'.format(prefix)]
    bbox_height = df['{}_bottom'.format(prefix)] - df['{}_top'.format(prefix)] 
    orig_width = bbox_width[bboxs_to_extend]
    orig_height = bbox_height[bboxs_to_extend]
    delta_width, delta_height, extended_width, extended_height = _extend_bounding_boxes(
                                        orig_width, orig_height, min_object_area)
    df.loc[bboxs_to_extend, '{}_left'.format(prefix)] -= delta_width / 2
    df.loc[bboxs_to_extend, '{}_right'.format(prefix)] = (
            df.loc[bboxs_to_extend, '{}_left'.format(prefix)] + extended_width)
    df.loc[bboxs_to_extend, '{}_top'.format(prefix)] -=  delta_height / 2
    df.loc[bboxs_to_extend, '{}_bottom'.format(prefix)] = (
            df.loc[bboxs_to_extend, '{}_top'.format(prefix)] + extended_height)
    df.loc[bboxs_to_extend, '{}_area'.format(prefix)] = extended_width * extended_height
    return df

def extend_detections_for_orig_ufo_based_on_area(df_results_orig_ufo, minimum_object_area):
    """This function extends detections of the original ufo algorithm to have 
    specified minimum area for all detections"""
    log.info('Extending small detections')
    is_small_det_area = df_results_orig_ufo['det_area'] < minimum_object_area 
    if len(df_results_orig_ufo[is_small_det_area]) == 0:
        log.info('There are no detections with area below %d', minimum_object_area)
        return df_results_orig_ufo
    log.info('Extending %d detections', len(df_results_orig_ufo[is_small_det_area]))
    df_results_orig_ufo =_extend_bounding_boxes_to_have_minimum_object_area(
            df_results_orig_ufo, 'det', is_small_det_area, minimum_object_area)
    min_det_area = df_results_orig_ufo[is_small_det_area]['det_area'].min()
    assert  min_det_area > minimum_object_area - 1, ('Something went wrong, '
                            'minimum detection area is still less then expected')
    return df_results_orig_ufo

def extend_bounding_boxes_based_on_gt_area(df_comb, minimum_object_area):
    """Extends ground truth and result bounding boxes based on the area of the ground truth: 
    If the area of ground truth bounding box is less than the minimum object area
    both ground truth and detection bounding boxes are extended to reach minimum object area, 
    while maintaining original aspect ratio 
    """
    log.info('Extending bounding boxes based on ground truth area')
    required_columns = DETECTION_BBOX_COLS_ + GROUNDTRUTH_BBOX_COLS_ + ['gt_area', 'det_area'] 
    required_columns_str = ', '.join(required_columns) 
    assert is_in_df_columns(df_comb, required_columns), (
        'One or more out of {} columns is not found, '
        'cannot perform bounding box extension'.format(required_columns_str))
    is_small_gt_area = (df_comb['gt_area'] > 0) & (df_comb['gt_area'] < minimum_object_area) 
    if len(df_comb[is_small_gt_area]) == 0:
        log.info('There are no objects with area below %d', minimum_object_area)
        return df_comb
    log.info('Number of objects with ground truth area less than %d is %d', 
                            minimum_object_area, len(df_comb[is_small_gt_area]))
    # extending ground truth bounding box for objects with small ground truth area
    df_comb = _extend_bounding_boxes_to_have_minimum_object_area(df_comb, 'gt', 
                                        is_small_gt_area, minimum_object_area) 
    # verify that all the boxes now have area no less than minimum object area
    assert df_comb['gt_area'].min() >= minimum_object_area - 1, (
                'Something went wrong, minimum ground truth area is still less '
                                                'then minimum expected area')

    # extending detection bounding box for objects with small ground truth area 
    # if the detection area is also small
    is_small_gt_and_det_area = np.logical_and(is_small_gt_area, 
                                (df_comb['det_area'] < (minimum_object_area - EPSILON_)))
    if len(df_comb[is_small_gt_and_det_area]) == 0:
        log.info('There are no detections with area below %d that are being matched '
                    'to extended ground truth', minimum_object_area)
        return df_comb

    log.info('Number of cases with ground truth and detection areas less '
        'than %d is %d', minimum_object_area, len(df_comb[is_small_gt_and_det_area]))

    df_comb = _extend_bounding_boxes_to_have_minimum_object_area(df_comb, 'det', 
                                is_small_gt_and_det_area, minimum_object_area) 
    # verify that all the detection boxes that are compared to extended ground 
    # truth boxes now have area no less than minimum object area
    assert df_comb[is_small_gt_area]['det_area'].min() > minimum_object_area - 1, (
        'Something went wrong, minimum detection area is still less then expected')
    return df_comb

def preprocess_results(df_res, minimum_object_area=None):
    """Add to the result a bounding box in top, left, bottom, right format and area
    """
    df_res = augment_with_detection_top_bottom_left_right(df_res)
    df_res = augment_with_detection_area(df_res)
    # extension of bounding boxes if necessary (typically done for original UFO only)
    return df_res

def _remove_invalid_groundtruth(df_gt):
    non_valid = df_gt['gt_area'] == 0
    cols = GROUNDTRUTH_BBOX_COLS_ 
    for col in cols + ['gt_area']:
        df_gt.loc[non_valid, col] = np.nan 
    return df_gt
    
def preprocess_groundtruth(df_gt):
    """Adds an area to ground truth bounding boxes
    """
    df_gt = augment_with_groundtruth_area(df_gt)
    df_gt = _remove_invalid_groundtruth(df_gt)
    return df_gt

def threshold_results_based_on_score(df_results, score_thresh):
    """Thresholds df_results based on the score"""
    assert is_in_df_columns(df_results, ['s']), (
        's (score) column is not found - cannot perform thresholding')
    df_results = df_results.query('s >= {}'.format(score_thresh))
    return df_results

def threshold_results_based_on_track_id_len(df_results, min_track_len):
    """helper to filter data frame of matches based on track id length so far"""
    min_track_len = min_track_len if min_track_len is not None else 0  
    if min_track_len < 0:
        raise ValueError('min_track_len should be positive or zero or None')
    elif min_track_len > 0:
        log.info('Filtering length of tracks so far = %.3f', min_track_len) 
        assert is_in_df_columns(df_results, ['track_id_len_so_far']), (
            'track_id_len_so_far column is not found, cannot filter')
        df_results = df_results.query('track_id_len_so_far == track_id_len_so_far and track_id_len_so_far >= {}'.format(min_track_len)) 
        return df_results
    return df_results

####################################################################################################
# MAIN
####################################################################################################
def get_matching_params(flags):
    is_match_thresh = flags.is_match_threshold
    is_no_match_thresh = flags.is_no_match_threshold
    if is_no_match_thresh is None:
        return is_match_thresh, is_match_thresh
    if is_match_thresh < is_no_match_thresh:
        raise ValueError('iou threshold for groundtruth and detection to be '
                            'declared as "no match" cannot be more than '
                            'iou threshold for a match')
    if is_no_match_thresh <= 0:
        raise ValueError('iou threshold for groundtruth and detection to be '
                            'declared as "no match" must be strictly positive')
    return is_match_thresh, is_no_match_thresh

def _assert_no_matches_if_not_both_gt_det_exist(df):
    """helper to check that matches is always 0 if detection or groundtruth do not exist 
    in the frame and no matches are always 1
    """ 
    required_columns = ['Exists', 'gt_det_match', 'gt_det_no_match']
    required_columns_str = ', '.join(required_columns) 
    assert is_in_df_columns(df, required_columns), (
        'One or more of {} is not found '.format(required_columns_str))
    df_match_error = df.query('Exists == "right_only" and gt_det_match != 0')
    df_no_match_error = df.query('Exists == "right_only" and gt_det_no_match != 1')
    assert len(df_match_error)== 0, 'match error, should be 0'
    assert len(df_no_match_error)== 0, 'no match error, should be 1'    

def augment_with_track_len(df):
    """calculates length of track and length of track so far"""
    df_unique_tracks_per_frame = df.groupby(['flight_id','track_id'])['frame'].agg(
                                             ['min', 'max', 'count']).add_prefix('track_frame_')
    df_track_id_len = df_unique_tracks_per_frame.reset_index(0).reset_index(0)
    df_track_id_len = df_track_id_len.assign(track_id_len = 
                     1 +  df_track_id_len['track_frame_max'] - df_track_id_len['track_frame_min'])
    # Sanity check for length of tracks
    min_track_id_len = df_track_id_len['track_id_len'].min()
    max_track_id_len = df_track_id_len['track_id_len'].max()
    assert min_track_id_len >= 0, 'Minimum track length: expected {}, got {}'.format(0, min_track_id_len)
    assert max_track_id_len <= MAX_FRAMES_PER_FLIGHT, (
            'Maximum track length: expected {}, got {}'.format(MAX_FRAMES_PER_FLIGHT, max_track_id_len))
    
    df = df.merge(df_track_id_len, on=['flight_id', 'track_id'], how='left')
    df = df.assign(track_id_len_so_far = 1 +  df['frame'] - df['track_frame_min'])
    assert len(df.query('track_id_len_so_far > track_id_len')) == 0, ('Track id len so far'
                                                    ' should not exceed total track id length ') 
    
    return df

def augment_with_zero_match_no_match(df):
    """this is a special case handling when no results are found"""
    df['gt_det_match'] = 0
    df['gt_det_no_match'] = 0
    return df

def compute_groundtruth_detections_matches(df_gt, df_results, extend_small_detections,
                     is_match_thresh, is_no_match_thresh=None,
                     minimum_object_area=0):
    """This function computes the matches between the ground truth and the detections 
    at the same frame
    Input:
        df_gt: pd.DataFrame - ground truth
        df_results: pd.Dataframe - detection results 
        extend_small_detections: Boolean - True if the detection results are derived from 
                        original UFO algorithm 
        is_match_thresh: float - threshold for matching function to determine correct 
                                match between ground truth and detection
        is_no_match_thresh: float - threshold that defines no match between 
                                    ground truth and detection
    Returns:
        df_comb_gt_results_outer_with_matches: pd.DataFrame - 
                        combined ground truth and detections with match / no match
    """    
    # pre-calculation of bounding boxes if necessary for matching
    df_gt = preprocess_groundtruth(df_gt)
    if extend_small_detections:
        # add detection bounding boxes and extend if necessary 
        df_results = preprocess_results(df_results, minimum_object_area)
    else:
        # add detection bounding boxes
        df_results = preprocess_results(df_results)

    # combine all ground truth intruders with all the detections withing the same img_name 
    # this pairs each ground truth intruder with each detection and vice versa
    log.info('Pairing each ground truth intruder with each detection in the respective frame')
    df_comb_gt_results = df_gt.merge(df_results, on =['flight_id', 'frame', 'img_name'], how='outer', 
                                    indicator='Exists', suffixes=['_gt', '_det'])
    
    assert len(df_comb_gt_results.query('Exists == "right_only"')) == 0, (
            'there are missing images in ground truth, that appear in detection file')
       
    log.info('Augmenting with original iou for comparison')
    df_comb_gt_results = augment_with_iou(df_comb_gt_results)        
    if minimum_object_area != 0: 
        # save the original iou to compare 
        df_comb_gt_results = df_comb_gt_results.rename(columns={'iou': 'iou_orig'})
        # extend bounding boxes to have all ground truth area equal to at least minimum area
        log.info('Extending bounding boxes based on groundtruth area')
        df_comb_gt_results = extend_bounding_boxes_based_on_gt_area(
                                df_comb_gt_results, minimum_object_area)
        log.info('Augmenting with extended iou with minimum object area of %d', minimum_object_area)
        df_comb_gt_results = augment_with_iou(df_comb_gt_results)
        assert np.all(df_comb_gt_results['iou'] - df_comb_gt_results['iou_orig']) >= 0, (
                                        'extended_iou should be higher or equal to original')            
    df_comb_gt_results_with_matches = augment_with_iou_match(
                        df_comb_gt_results, is_match_thresh, is_no_match_thresh)
    _assert_no_matches_if_not_both_gt_det_exist(df_comb_gt_results_with_matches)
    log.info('Matching done')
    return df_comb_gt_results_with_matches

def run(flags):  
    # preparing path for saving results 
    if flags.output_dir_path is None:
        # create a directory with the same name as airborne_classifier_result omitting the extension 
        output_dir = flags.airborne_classifier_results
        for extension in ['.json', '.csv', '.gz']:
            output_dir = output_dir.replace(extension, '')
        output_dir += '_metrics_min_track_len_{}'.format(flags.min_track_len)
    else:
        output_dir = flags.output_dir_path

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    is_match_threshold, is_no_match_threshold = get_matching_params(flags)
    matching_alg_str = 'extended_iou'
    prefix = flags.results_name_prefix
    matching_alg_str += '_minObjArea_{}'.format(flags.minimum_object_area)
    if flags.extend_small_detections:
        prefix = prefix.replace('det', 'ext_det')
    output_filename = ('{}_matches_{}_matchThresh_{}_noMatchThresh_{}.csv'.format(prefix,
                      matching_alg_str, str(is_match_threshold).replace('.','_'),
                                    str(is_no_match_threshold).replace('.','_')))
    full_output_path = os.path.join(output_dir, output_filename)
    
    # Starting processing
    log.info('Reading input ground truth and results')                    
    df_gt = get_deeplearning_groundtruth_as_data_frame(flags.deeplearning_groundtruth)
    if (flags.deeplearning_groundtruth.endswith('.json') 
        or flags.deeplearning_groundtruth.endswith('.json.gz')):
        log.info('Saving groundtruth in .csv format, please use .csv in the future') 
        df_gt.to_csv(flags.deeplearning_groundtruth.replace('.json', '.csv').replace('.gz', ''))
    log.info('Number of evaluated images is %d', df_gt['img_name'].nunique())
    df_results = get_results_as_data_frame(flags.airborne_classifier_results)
 
    if (flags.airborne_classifier_results.endswith('.json') 
        or flags.airborne_classifier_results.endswith('.json.gz')):
        log.info('Saving airborne classifier results in .csv format, please use .csv in the future') 
        df_results.to_csv(flags.airborne_classifier_results.replace('.json', '.csv').replace('.gz', ''))
    log.info('Number of evaluated unique detections is %d', len(df_results))
    log.info('Filtering results based on results score %.2f', flags.detection_score_threshold)  
    df_results = threshold_results_based_on_score(df_results, flags.detection_score_threshold)
    # enumerate detections with unique ids 
    df_results= df_results.sort_values('img_name').reset_index(0)

    if 'detection_id' not in df_results.columns:
        log.info('Enumerating detections with detection_id')
        df_results = augment_with_detection_id(df_results)
    # add track_id/ object_id
    if 'track_id' not in df_results.columns:
        if 'object_id' in df_results.columns:
            df_results = df_results.assign(track_id = df_results['object_id'])
            log.info('Using object_id as track_id')
        else:
            df_results = df_results.assign(track_id = df_results['detection_id'])
            log.info('Using detection_id as track_id')
    else:
        log.info('Using track_id as track_id')
    df_results = df_results.merge(df_gt[['flight_id','frame','img_name']].drop_duplicates(), 
                                                                on='img_name', how='left')
    # TODO: remove below when Sleipnir dataset is fixed to have 306 flights
    df_results = df_results.dropna(subset=['flight_id']) 
    log.info('Augmenting with track length')
    df_results = augment_with_track_len(df_results)
    log.info('Filtering results with track length below {}'.format(flags.min_track_len))
    df_results = threshold_results_based_on_track_id_len(df_results, flags.min_track_len)

    log.info('Computing ground truth and detection match based on %s', matching_alg_str)
    df_comb_gt_results_with_matches = compute_groundtruth_detections_matches(
        df_gt, df_results, flags.extend_small_detections,  
        is_match_threshold, is_no_match_threshold, flags.minimum_object_area)

    # save provided ground truth with the added encounters as data frame in .csv format
    log.info('Saving ground truth and detection match results to %s', full_output_path)
    df_comb_gt_results_with_matches.to_csv(full_output_path) 
    return full_output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates matches '
                                         'between the ground truth and results')
    add_flags(parser)
    parser.add_argument('--log-level', default=logging.getLevelName(logging.INFO), 
                                                help='Logging verbosity level')
    args = parser.parse_args()
    setup_logging(args.log_level)
    check_flags(args)
    run(args)    


