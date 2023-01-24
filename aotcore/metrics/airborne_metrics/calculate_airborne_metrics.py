# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""This module computes airborne level metrics.
The module assumes the following inputs available :
1) Ground truth enriched with encounters data frame (artifact of calculate_encounters.py)
2) Ground truth vs. Detection matches data frame (artifact of match_groundtruth_results.py)
3) Below/Mixed/Above horizon indicator for encounter (Below == -1, Above == 1, Mixed  = (-1,1))
The module saves an artifact that provides detection information for each encounter:
1) The range at which the encounter was detected 
2) The latency (in frames) it took to detect the encounter
""" 

import argparse
import json
import logging
import os
from functools import partial
from collections import  Counter, OrderedDict

import numpy as np
import pandas as pd

from .calculate_encounters import exclude_encounters_from_evaluation, DEFAULT_MIN_DETECTION_RANGE_M, ENC_END_RANGE_SCALE 
from .pandas_utils import is_in_df_columns 
from .script_utils import setup_logging

RESULTS_NAME_PREFIX_ = 'airborne_metrics'
DEFAULT_MIN_DETECTION_SCORE_ = 0.0 # minimum possible score of detection algorithm
DEFAULT_MAX_DETECTION_SCORE_ = 1.0 # maximum possible score of detection algorithm
DEFAULT_TEMPORAL_WINDOW_FOR_FL_DR_ = 30 # frames to consider for frame level detection rate calculation
DEFAULT_MIN_FL_DR_ = 0.5 # that means that within 30 frames at least 15 should be detected
DEFAULT_METRICS_VALUE_PRECISION_ = 5 # required for binary search for target value of target metrics
SCORE_PRECISION_ = 0.00001 # required for binary search for target value of target metrics
MINS_PER_FLIGHT = 2 # each flight has two minutes 
MINS_TO_HOURS = 1 / 60 
TARGET_FAR_PER_HOUR = 1 / 2 # 1 False Alarm per 2 hours
RANGES_TO_DETECT = [300] # the metrics is reported for detections before 300 m
FAST_DETECTION_WINDOW_ = 30 # frames - an encounter is detected fast if it is detected within 3 secs.

# Frame level detection rate calculation constants
DEFAULT_MIN_OBJECT_AREA = 200 # minimum object area for frame level 
NOT_BIRD_QUERY = 'id.str.contains("Flock") == False and id.str.contains("Bird") == False'
GT_ABOVE_AREA_QUERY = 'gt_area > {min_area}'
GT_BELOW_AREA_QUERY = 'gt_area <= {min_area}'
DEFAULT_OBJECT_OF_INTEREST_QUERY = GT_ABOVE_AREA_QUERY + ' and ' + NOT_BIRD_QUERY

# QUERIES 
HORIZON_QUERIES = {'Below Horizon': 'is_above_horizon == -1', 
                   'Mixed': '-1 < is_above_horizon < 1',
                   'Above Horizon': 'is_above_horizon == 1',
                   'All': 'is_above_horizon == is_above_horizon'}
FALSE_DETECTION_QUERY = 'gt_det_no_match == 1'
POSITIVE_DETECTION_QUERY = 'gt_det_match == 1'
PLANNED_INTRUDERS = 'range_distance_m == range_distance_m' # that is range is a number (not NaN)
NON_PLANNED_INTRUDERS = 'range_distance_m != range_distance_m' # that is range is NaN
NON_PLANNED_AIRCRAFT = NON_PLANNED_INTRUDERS + ' and ' + NOT_BIRD_QUERY


log = logging.getLogger(__name__)

def add_flags(parser):
    """Utility function adding command line arguments to the parser"""
    # input files 
    parser.add_argument('--groundtruth-results-matches-filename', '-m', required=True, 
                        help='Path to the ground truth and detection matches data frame in .csv format')
    parser.add_argument('--encounters-with-groundtruth-filename', '-e', required=True, 
                        help='Path to the ground truth enriched with encounters data frame in .csv format')
    # output  
    parser.add_argument('--output-dir-path', '-o',
                        help='Desired folder to save the output data frame with '
                        'match/no match between groundtruth and detections')
    parser.add_argument('--results-name-prefix', type=str, default=RESULTS_NAME_PREFIX_,
                        help='Prefix for results filename')
    parser.add_argument('--save-intermediate-results', default=False, action='store_true',
                        help='Specify this if saving intermediate data frame with encounters and '
                        'corresponding moving frame level detection rate is needed')
    # working point parameters 
    parser.add_argument('--min-object-area', type=float, default=DEFAULT_MIN_OBJECT_AREA,
                        help='The minimum object area for average frame level detection rate calculation')
    parser.add_argument('--min-enc-range-upper-bound', type=float,  
                        default=ENC_END_RANGE_SCALE * DEFAULT_MIN_DETECTION_RANGE_M,
                        help='The minimum range of the encounter should be not less than this value, '
                        'default is {}'.format(ENC_END_RANGE_SCALE * DEFAULT_MIN_DETECTION_RANGE_M))
    parser.add_argument('--max-enc-range-lower-bound', type=float, 
                        help='The maximum range of the encounter should be not less than this value')
    parser.add_argument('--target-metrics', '-t', type=str, choices=['far', 'fppi', 'fl_dr'], 
                       help='Provide metrics, FAR or FPPI or FL_DR (frame-level detection rate), '
                       'to determine a working point. This is useful when comparing to other algorithms. '
                       'If None is provided, detection score threshold (default = 0) will be used')
    parser.add_argument('--target-value', '-f', type=float, 
                       help='Provide the value for the expected target metrics (if chosen). '
                       'The default target values is calculated if target metrics is FAR. '
                       'If target metrics is FPPI or FL_DR and target values is None - error will be thrown')
    parser.add_argument('--target-value-precision', type=int, default=DEFAULT_METRICS_VALUE_PRECISION_,
                       help='Precision with which to calculate targeted value. Provide this value '
                       'if you want the metrics to calculate the score based on specific target metrics')
    parser.add_argument('--min-det-score', type=float, default=DEFAULT_MIN_DETECTION_SCORE_,
                        help='Minimum possible detection score. Provide this value if you want '
                       'the metrics to calculate the score for working point based on target metrics')
    parser.add_argument('--max-det-score', type=float, default=DEFAULT_MAX_DETECTION_SCORE_,
                        help='Maximum possible detection score. Provide this value if you want '
                       'the metrics to calculate the score for working point based on target metrics')
    parser.add_argument('--detection-score-threshold', '-s', type=float, 
                        default=DEFAULT_MIN_DETECTION_SCORE_,
                        help='Detection score threshold for working point')
    # parameters 
    parser.add_argument('--use-track-fl-dr', action='store_true',
                        help='Setting up this flag will require the same track_id in detections that ' 
                        'contribute to the encounter level detection rate calculation')
    parser.add_argument('--fl-dr-temporal-win', type=int, default=DEFAULT_TEMPORAL_WINDOW_FOR_FL_DR_,  
                        help='Temporal window for moving frame level detection rate')
    parser.add_argument('--min-fl-dr', type=float, default=DEFAULT_MIN_FL_DR_,
                         help='Minimum frame level detection rate within the temporal window')

def _assert_non_negative(value, value_name):
    """assertion helper"""
    assert value >= 0.0, ('{} is expected to be non-negative'.format(value_name))

def _assert_strictly_positive(value, value_name):
    """assertion helper"""
    assert value > 0.0, ('{} is expected to be strictly positive, but received {}'.format(
                                                                        value_name, value))

def check_flags(flags):
    """Utility function to check the input"""
    _assert_non_negative(flags.min_det_score, 'Minimum detection score')
    _assert_non_negative(flags.max_det_score, 'Maximum detection score')
    _assert_non_negative(flags.detection_score_threshold, 'Detection score threshold')
    assert (flags.target_value is not None or flags.target_metrics is None 
        or flags.target_metrics == 'far'), (
                'If target-metrics is specified as fppi or fl_dr, target-value should be provided ')
    if flags.target_value is not None:
        _assert_non_negative(flags.target_value, 'Target value')
        _assert_strictly_positive(flags.target_value_precision, 'Target value precision')
    _assert_strictly_positive(flags.fl_dr_temporal_win, 
                                'Temporal window for moving frame level detection rate')
    assert 0 < flags.min_fl_dr <= 1.0, (
                        'Minimum frame level detection rate should be in (0,1] range')
    assert flags.output_dir_path is None or not os.path.isfile(flags.output_dir_path), (
                                                 'Directory name is expected as output path')
    assert (flags.groundtruth_results_matches_filename.endswith('.csv') and 
            flags.encounters_with_groundtruth_filename.endswith('.csv')), (
            'Unsupported file format, please provide .csv produced by calculate_encounters.py and '
            'match_groundtruth_results.py')

##########################################################
# Frame-level FPPI and PD and strict FAR  calculation code
##########################################################

def _calc_num_no_match_detections(df_matches):
    """helper to calculate number of not matched detection within the provided data frame of matches"""
    log.info('Calculating the number of detections that did not match ground truth') 
    if not is_in_df_columns(df_matches, ['detection_id']):
        # this is an edge case when there are 0 detections
        return 0
    assert is_in_df_columns(df_matches, ['detection_id', 'gt_det_no_match']), (
        'One or more of detection_id, gt_det_no_match columns is not found, cannot calculate')
    # Grouping by 'detection_id' yields all the match results between the detection with 
    # this specific id and a group of ground truth intruders that were evaluated for a match 
    # (within the same frame).
    # gt_det_no_match will be 1 if the detection does not match the specific intruder
    # if minimum of 'gt_det_no_match' over the group of intruders equals 1, it means that the 
    # detection did not match any of those intruders, hence it is a false detection    
    df_detections = df_matches[['detection_id', 'gt_det_no_match']].groupby(
                                            'detection_id')['gt_det_no_match'].min().reset_index(0)
    number_of_detections = df_matches['detection_id'].nunique()
    assert number_of_detections == len(df_detections), ('something went wrong with grouping detections, '
                        'expected {}, but got {}'.format(number_of_detections, len(df_detections)))

    num_no_match_detections = len(df_detections.query(FALSE_DETECTION_QUERY)) 
    log.info('No match calculation: Number of detections without a match = %d out of %d '
            'unique detections', num_no_match_detections, number_of_detections)
    return num_no_match_detections

def _calc_num_unique_track_ids_with_no_match_detection(df_matches):
    """helper to calculate number of unique tracks that correspond to at least one not matched detection"""
    log.info('Calculating the number of unique tracks ids that that correspond to at least one '
            'not matched detection') 
    if not is_in_df_columns(df_matches, ['detection_id']):
        # this is an edge case when there are 0 detections
        return 0
    required_cols = ['flight_id', 'detection_id', 'track_id', 'gt_det_no_match']
    assert is_in_df_columns(df_matches, required_cols), (
        'One or more of detection_id, track_id, gt_det_no_match columns is not found, cannot calculate')
    
    # Grouping by 'detection_id' yields all the match results between the detection with 
    # this specific id and a group of ground truth intruders that were evaluated for a match 
    # (within the same frame).
    # gt_det_no_match will be 1 if the detection does not match the specific intruder
    # if minimum of 'gt_det_no_match' over the group of intruders equals 1, it means that the 
    # detection did not match any of those intruders, hence it is a false detection 
    # Only unique track_ids that correspond to false detection are counted    
    df_detections = df_matches[required_cols].groupby('detection_id')[
                ['flight_id', 'track_id', 'gt_det_no_match']].agg({
                'flight_id':'first', 'track_id': 'first', 'gt_det_no_match': 'min'}).reset_index(0)
    num_false_tracks = df_detections.query(FALSE_DETECTION_QUERY).groupby(['flight_id', 'track_id']).ngroups
    log.info('Number of unique track_ids that correspond to at least one false detection %d', 
                                                                                    num_false_tracks)
    return num_false_tracks

def _filter_matches_based_on_detection_score(df_matches, min_score):
    """helper to filter data frame of matches based on detection score"""
    min_score = min_score if min_score is not None else 0    
    if min_score < 0:
        raise ValueError('min_score should be positive or zero or None')
    elif min_score > 0:
        log.info('Filtering score threshold = %.3f', min_score) 
        assert is_in_df_columns(df_matches, ['s']), 's (score) column is not found, cannot filter'
        df_matches_filtered = df_matches.query('s >= {}'.format(min_score))
        return df_matches_filtered
    return df_matches

def compute_false_positives_per_image(df_matches, total_frames_processed=None, min_score=None):
    """Compute FPPI based on a data frame of matches - useful for frame-level metrics 
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        total_frames_processed: int, number of processed flights if different from the number of unique
                frames ('img_name') in the provided df_matches
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, false positives per image
    """
    # determine how many images were processed 
    if total_frames_processed is not None and total_frames_processed <= 0:
        raise ValueError('total_frames_processed should be strictly positive')
    if total_frames_processed is None:      
        log.info('FPPI calculation: Using unique image names in the provided data frame to calculate '
                                                                'total number of processed frames')
        assert is_in_df_columns(df_matches, ['img_name']), 'img_name column is not found, cannot calculate'
        total_frames_processed = df_matches['img_name'].nunique()
    log.info('FPPI calculation: Total number of processed frames is %d', total_frames_processed)
    df_matches = _filter_matches_based_on_detection_score(df_matches, min_score)
    fppi = _calc_num_no_match_detections(df_matches) / total_frames_processed
    log.info('FPPI = %.5f', fppi)
    return fppi

def compute_false_alarms_per_hour(df_matches, total_flights_processed=None, min_score=None):
    """Compute strict FAR based on a data frame of matches, based on the following definition
    Overall False Alarm Rate (strict FA) - a number of unique reported track ids, 
    which correspond to at least one false positive cluster, divided by total number of hours 
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        total_flights_processed: int, number of processed flights if different from the number of unique
                flights ('flight_id') in the provided df_matches
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, false alarms per hour
    """
    # determine how many images were processed 
    if total_flights_processed is not None and total_flights_processed <= 0:
        raise ValueError('total_flights_processed should be strictly positive')
    if total_flights_processed is None:      
        log.info('FAR calculation: Using unique flight ids in the provided data frame to calculate '
                                                                'total number of processed flights')
        assert is_in_df_columns(df_matches, ['flight_id']), 'flight_id column not found, cannot calculate'
        total_flights_processed = df_matches['flight_id'].nunique()
    total_hours_processed = total_flights_processed * MINS_PER_FLIGHT * MINS_TO_HOURS
    _assert_strictly_positive(total_flights_processed, 'Total processed hours')
    log.info('FAR calculation: Total number of processed flights is %d', total_flights_processed)    
    log.info('FAR calculation: Total number of processed hours is %.3f', total_hours_processed)
   
    df_matches = _filter_matches_based_on_detection_score(df_matches, min_score)

    num_false_tracks =_calc_num_unique_track_ids_with_no_match_detection(df_matches)
    far = num_false_tracks / total_hours_processed
    log.info('FAR = %.5f', far)
    return far

def _calc_num_detected_intruders(df_matches):
    """helper to calculate number of detected intruders"""
    log.info('Calculating the number of intruders that were matched by detections') 
    if not is_in_df_columns(df_matches, ['detection_id']):
        # this is an edge case when there are 0 detections
        return 0
    assert is_in_df_columns(df_matches, ['detection_id', 'id', 'gt_det_match']), (
        'One or more of detection_id, id, gt_det_match columns is not found, cannot calculate')
    
    # When grouping by 'img_name', 'id' we get all the match results between the specific 
    # intruder/object (within specific frame) and a group of detections that were evaluated for a match 
    # "gt_det_match" will be 1 if a detection does match the specific intruder
    # if maximum of 'gt_det_match' over the group of detections equals 1, it means that the at least
    # one detection matched the intruder and hence the intruder is detected 
    df_intrudes = df_matches[['img_name', 'id', 'gt_det_match']].groupby(
                                    ['img_name', 'id'])['gt_det_match'].max().reset_index(0)
    num_detected_intruders = len(df_intrudes.query(POSITIVE_DETECTION_QUERY)) 
    log.info('Detected intruders calculation: Number of detected intruders = %d ', num_detected_intruders)
    return num_detected_intruders

def compute_probability_of_detection(df_matches, min_score=None):
    """Compute frame-level PD of valid intruders - useful for frame-level metrics.
    This function does NOT assume planned intruders 
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, frame-level probability of detection
    """
    tot_objects_to_detect = df_matches.groupby(['img_name', 'id']).ngroups
    if tot_objects_to_detect == 0:
        return 0
    log.info('PD calculation: Number of intruders to detect = %d', tot_objects_to_detect)

    df_matches = _filter_matches_based_on_detection_score(df_matches, min_score)

    # each intruder/object is identified by img_name and id 
    # gt_det_match will be 1 if there is match between some detection and this object
    # grouping by 'img_name', 'id' will create a group with all the matches to all the detections
    # in this frame (img_name) for this specific object 
    # if there is at least one detection that matches this objects the maximum of gt_det_match equals 1
    num_matched_objects = _calc_num_detected_intruders(df_matches) 
    pd = num_matched_objects / tot_objects_to_detect
    log.info('PD = %.3f = %d / %d', pd, num_matched_objects, tot_objects_to_detect)
    return pd, num_matched_objects, tot_objects_to_detect

def _get_planned_intruders_in_range(df_matches, min_range, max_range=None):
    """helper to get only planned intruders in range"""
    if max_range is not None:
        assert is_in_df_columns(df_matches, ['range_distance_m']), (
                            'range_distance_m column is not found - cannot filter based on range')
        df_matches_in_range = df_matches.query(
                                        '{} <= range_distance_m <= {}'.format(min_range, max_range))
    else:
        # evaluate against planed intruders only
        df_matches_in_range = df_matches.query('range_distance_m != range_distance_m')
    return df_matches_in_range

def compute_probability_of_detection_of_planned_intruders(df_matches, max_range=None, min_range=0,
                                                         min_score=None):
    """Compute frame-level PD of PLANNED intruders - useful for frame-level metrics.
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        max_range: float, maximum range of intruder to consider for evaluation, if not provided all
                planned intruders with valid range are used 
        min_range: float, minimum range of intruder to consider for evaluation, if not provided all
                planned intruders with valid range are used 
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, probability of detection
    """
    log.info('PD calculation: Intruders Range =  [%.1f, %.1f]', min_range, max_range)
    df_matches_in_range = _get_planned_intruders_in_range(df_matches, min_range, max_range)  
    return compute_probability_of_detection(df_matches_in_range, min_score)

def compute_probability_of_detection_small_objects(df_matches, min_area=DEFAULT_MIN_OBJECT_AREA, 
                                    min_score=None, obj_query = DEFAULT_OBJECT_OF_INTEREST_QUERY):
    """Compute frame-level PD of PLANNED intruders - useful for frame-level metrics.
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results 
                (typically this is an artifact of match_groundtruth_results.py)
        min_area: float, minimum area of intruder to consider for evaluation, default is 300
        min_score: float, minimum detection score to consider for evaluation 
    Returns:
        float, probability of detection

    """
    aircraft_area_query = obj_query.format(min_area=min_area)
    log.info('PD calculation: %s', aircraft_area_query)
    df_matches_area = df_matches.query(aircraft_area_query)
    return compute_probability_of_detection(df_matches_area, min_score)

##########################################################
# Airborne-level PD calculation code
##########################################################

def get_valid_encounters(df_encounters):
    return df_encounters.query('is_valid_encounter == True')

def combine_encounter_with_gt_det_matches(df_matches, df_encounters):
    """Combines two data frames based on encounter identifier
    Parameters:
        df_matches: pd.DataFrame, data frame of matches between ground truth and detection results
        df_encounters: pd.DataFrame, data frame with ground truth and and encounter info
    Returns:
        pd.DataFrame, combined data frame
    """
    cols_to_combine_on = ['flight_id', 'img_name', 'frame', 'id']
    required_columns_str = ', '.join(cols_to_combine_on) 
    assert is_in_df_columns(df_matches, cols_to_combine_on), (
        'One or more out of {} columns is not found in data frame of matches, '
                                'cannot combine'.format(required_columns_str))
    assert is_in_df_columns(df_encounters, cols_to_combine_on), (
        'One or more out of {} columns is not found in data frame of encounters, '
                                'cannot combine'.format(required_columns_str))
    df = df_encounters.merge(df_matches, on=cols_to_combine_on, 
            how='left', suffixes=['_orig', '']).sort_values(by =['encounter_id','img_name', 'frame'])
    return df

def augment_with_moving_frame_level_detection_rate_per_encounter(df_matches, temporal_window):
    """adds moving frame level detection rate per encounter, based on the provided temporal_window"""

    required_columns_str = ', '.join(['encounter_id', 'gt_det_match']) 
    assert is_in_df_columns(df_matches, ['encounter_id', 'gt_det_match']), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    df_matches = df_matches.fillna(value={'gt_det_match': 0})
    df_with_moving_fl_dr = df_matches.groupby(['encounter_id'])['gt_det_match'].rolling(
                                                temporal_window).apply(np.mean).reset_index()
    df_with_moving_fl_dr = df_with_moving_fl_dr.rename(columns={'gt_det_match': 'fl_dr'})
    return df_with_moving_fl_dr

def augment_with_moving_most_common_track_id_count_per_encounter(df_matches, temporal_window):
    """adds moving frame level detection rate per encounter, based on the provided temporal_window"""

    required_columns_str = ', '.join(['encounter_id', 'matched_track_id']) 
    assert is_in_df_columns(df_matches, ['encounter_id', 'matched_track_id']), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    
    def get_most_common_freq(all_track_ids): 
        all_track_ids_list = []
        for element in all_track_ids:
            all_track_ids_list.extend(element)
        track_id_counter = Counter(all_track_ids_list)
        most_common = track_id_counter.most_common(2) # taking 2 most common track ids 
                                        # 2 because one can be -1, and is not counted for detection
        if  most_common[0][0] != -1:
            most_common_freq = (most_common[0][1] / temporal_window) 
        else:
            if len(most_common) > 1:
                most_common_freq = (most_common[1][1] / temporal_window) 
            else:
                most_common_freq = 0
        return most_common_freq

    def my_rolling_apply_char(frame, window, func):
        index = frame.index
        values = [0 if i + 1 - window < 0 else func(frame.iloc[i + 1 - window : i + 1]) for 
                                                                        i in range(0, len(frame))]
        return pd.DataFrame(data={'track_fl_dr': values}, index=index).reindex(frame.index)

    df_same_track_id_count = df_matches.groupby(['encounter_id'])
    
    df_matched_track_freq = pd.DataFrame(columns=['encounter_id', 'track_fl_dr'])
    for encounter_name, group in df_same_track_id_count: 
        df_res = my_rolling_apply_char(group['matched_track_id'], temporal_window, get_most_common_freq)
        df_res = df_res.assign(encounter_id = encounter_name)
        df_matched_track_freq = df_matched_track_freq.append(df_res)
    df_matched_track_freq.index.name = 'frame'
    return df_matched_track_freq.reset_index()

def augment_with_diff_to_first_frame(df_encounters):
    """adds difference in frame between each frame in encounter and the first frame of the encounter"""
    required_columns_str = ', '.join(['frame', 'framemin']) 
    assert is_in_df_columns(df_encounters, ['frame', 'framemin']), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))

    diff_to_first_frame = df_encounters['frame'] - df_encounters['framemin']
    df_encounters = df_encounters.assign(delta_to_min_frame = diff_to_first_frame)
    return df_encounters

def augment_with_detection_info(df_encounters_info, fl_dr_thresh, use_track_fl_dr=False):
    """adds maximum moving frame level detection rate to each encounter and if it its above
    or equal to the provided threshold the detection range and latency are added""" 
   
    fl_dr_col = 'fl_dr'
    if use_track_fl_dr:
        fl_dr_col = 'track_' + fl_dr_col
    def calc_detection_info(df):
        detection_range = np.nan
        det_latency = np.nan
        max_fl_dr = df[fl_dr_col].max() 
        if max_fl_dr >= fl_dr_thresh:
            first_index_above_thresh = df[df[fl_dr_col].ge(fl_dr_thresh, fill_value=0)].index[0]
            detection_range = df['range_distance_m'][first_index_above_thresh]
            det_latency = df['delta_to_min_frame'][first_index_above_thresh]
        return pd.Series(data=[max_fl_dr, detection_range, det_latency], 
                        index=['max_fl_dr', 'det_range_m', 'det_latency_frames'])
    required_cols = ['encounter_id', 'range_distance_m', 'delta_to_min_frame', fl_dr_col]
    required_columns_str = ', '.join(required_cols) 
    assert is_in_df_columns(df_encounters_info, required_cols), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    df_enc_det_info = df_encounters_info.groupby(['encounter_id'])[
                ['range_distance_m', fl_dr_col, 'delta_to_min_frame']].apply(calc_detection_info)    
    return df_enc_det_info

def compute_moving_frame_level_detection_rate_per_encounter(df_matches, df_val_encounters, 
                                                min_score, fl_dr_temporal_win, use_track_fl_dr=False):
    """Computes moving frame level detection rate per encounter. The detection matches 
    are counted within the provided fl_dr_temporal_win, which slides across the frames that belong
    to the encounter. Detections with score less then min_score are filtered. 
    The detection rate is calculated only for valid encounters.
    """
    required_cols = ['s', 'flight_id', 'img_name', 'frame', 'id', 'gt_det_match']
    if use_track_fl_dr:
        required_cols += ['track_id']
    required_columns_str = ', '.join(required_cols) 
    assert is_in_df_columns(df_matches, required_cols), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    assert is_in_df_columns(df_val_encounters, ['encounter_id']), (
        'encounter_id column is not found, cannot augment')
    
    def flatten(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
        return list_of_lists[:1] + flatten(list_of_lists[1:]) 
    
    log.info('Thresholding score')
    below_thresh_det = df_matches['s'] < min_score
    df_matches.loc[below_thresh_det, 'gt_det_match'] =  0 
    df_intruders_matches = df_matches.groupby(
            ['flight_id', 'img_name', 'frame', 'id'], as_index=False)['gt_det_match'].max()
    
    num_encounters =  df_val_encounters['encounter_id'].nunique()
    log.info('Number of encounters to detect %d', num_encounters)
    df_val_encounters = augment_with_diff_to_first_frame(df_val_encounters)
    log.info('Combining encounters with results')

    df = combine_encounter_with_gt_det_matches(df_intruders_matches, df_val_encounters)
    
    log.info('Grouping data frame with matches to getdetection matches per encounter')
    df_encounters_with_frame_matches = df.groupby(
                                    ['encounter_id','frame'])['gt_det_match'].max().reset_index(0)
    log.info('Augmenting with moving frame level detection rate, this might take some time')
    df_with_moving_fl_dr = augment_with_moving_frame_level_detection_rate_per_encounter(
                                        df_encounters_with_frame_matches, fl_dr_temporal_win)
    
    log.info('Merge frame_level detection rate ')
    df = df.merge(df_with_moving_fl_dr, on=['encounter_id','frame'], how='left')    

    if use_track_fl_dr:
        df_matches = df_matches.assign(matched_track_id = [track_id if is_match else  -1 
                for track_id, is_match in zip(df_matches['track_id'], df_matches['gt_det_match'])])
        log.info('Grouping data frame with matches to get matched track_ids per frame and object')
        df_matched_track_ids = df_matches.groupby(['flight_id', 'img_name', 'frame', 'id'], 
                                as_index=False)['matched_track_id'].agg(lambda x: flatten(list(x)))

        df2 = combine_encounter_with_gt_det_matches(df_matched_track_ids, df_val_encounters)
        log.info('Grouping data frame with matches to get matched track_ids per encounter and frame')
        df_encounters_with_matched_track_ids = df2.groupby(
            ['encounter_id','frame'])['matched_track_id'].agg(lambda x: flatten(list(x))).reset_index(0)
        df_with_track_id_count = augment_with_moving_most_common_track_id_count_per_encounter(
                                            df_encounters_with_matched_track_ids, fl_dr_temporal_win)
        df2 = df2.merge(df_with_track_id_count, on=['encounter_id','frame'], how='left')
        df = df.merge(df2[['encounter_id','frame', 'matched_track_id', 'track_fl_dr']], 
                                            on=['encounter_id','frame'], how='left')
    # asserting correctness of detection rate calculation based on track_id compared to regular fl_dr
    assert is_in_df_columns(df, ['fl_dr', 'track_fl_dr']), 'fl_dr or track_fl_dr not found'
    df_wrong = df.query('fl_dr < track_fl_dr') 
    assert len(df_wrong) == 0, 'track frame level detection rate is wrong'  
    return df

def get_encounter_frame_level_info(df):
    """Provides basic information about encounters:
    encounter_id - encounter id 
    flight_id - which flight it belongs to 
    framemin - first frame of the encounter 
    framemax - last frame of the encounter 
    tot_num_frames - total number of frames in encounters (without gaps)
    num_matched_frames  - number of frames within encounter with matched ground truth
    is_above_horizon - 1 - above horizon, 0 - below horizon, any value in (-1, 1) is mixed 
    """
    required_cols = ['encounter_id', 'flight_id', 'framemin', 'framemax','framecount', 'gt_det_match', 
                    'frame', 'is_above_horizon']
    required_columns_str = ', '.join(required_cols) 
    assert is_in_df_columns(df, required_cols), (
        'One or more out of {} columns is not found, cannot augment'.format(required_columns_str))
    
    df_partial= df[required_cols]
    df_encounter_frame_level_info = df_partial.groupby('encounter_id').agg(
                {'flight_id': 'first', 'framemin': 'first', 'framemax': 'first', 'framecount': 'first', 
                'frame': 'count', 'gt_det_match': 'sum', 'is_above_horizon': 'mean'}).reset_index(0)    

    df_encounter_frame_level_info = df_encounter_frame_level_info.rename(columns={
                                                            'frame': 'tot_num_frames', 
                                                            'gt_det_match': 'num_matched_frames'})
    assert len(df_encounter_frame_level_info.query('tot_num_frames != framecount')) == 0, (
            'something went wrong frame counts do not agree')
    
    return df_encounter_frame_level_info

def _augment_with_is_encounter_detected_before_range(df_encounter_detections, 
                                                    temporal_win, ranges_to_detect):
    """ helper to figure out if encounter is detected based on detection range and latency"""
    is_detected_fast = df_encounter_detections['det_latency_frames'] < FAST_DETECTION_WINDOW_ #frames
    for range_to_detect in ranges_to_detect:
        is_detected_before_this_range = df_encounter_detections['det_range_m'] >= range_to_detect
        df_encounter_detections = df_encounter_detections.assign(detected_before = 
                                1 * np.logical_or(is_detected_before_this_range, is_detected_fast))
        df_encounter_detections = df_encounter_detections.rename(columns={'detected_before':
                                                    'detected_before_{}'.format(range_to_detect)})
    return df_encounter_detections


def compute_encounter_detections(df_enc_with_fl_dr, min_frame_level_dr, temporal_win, 
                                    ranges_to_detect=RANGES_TO_DETECT, use_track_fl_dr=False):
    """Computes if encounter was detected based on provided function is_encounter_detected_func
    Params:
        df: pd.DataFrame, data frame with encounters and matches of detection to intruders info
        min_frame_level_dr: float, minimum frame level rate required for encounter
        ranges_to_detect: List[float], list of ranges at which to evaluate detection
    Returns:
        pd.DataFrame of encounters with information on detection per encounter   
    """ 
    assert is_in_df_columns(df_enc_with_fl_dr, ['encounter_id']), ('encounter_id not in data frame, '
                                                    'cannot calculate encounter detections')
    log.info('Checking if encounters were detected')
    df_encouter_detection_info = augment_with_detection_info(df_enc_with_fl_dr, min_frame_level_dr, 
                                                                                use_track_fl_dr)
    df_encounter_frame_level_info = get_encounter_frame_level_info(df_enc_with_fl_dr)      
    df_encounter_info = df_encounter_frame_level_info.merge(df_encouter_detection_info, 
                                                            on='encounter_id', how='left')
    df_encounter_detections = _augment_with_is_encounter_detected_before_range(df_encounter_info, 
                                                                    temporal_win, ranges_to_detect)
    return df_encounter_detections

def search_score_for_target_func(min_score, max_score, target_func, target_func_val, 
                                target_func_val_precision):
    """This function performs a search for a score that receives a certain value of the provided 
    target function. The search is done using binary search. There is an assumption that the function
    is monotonical with respect to the scores. The direction is determined based on minimum and middle
    scores
    Parameters:
        min_score: float, minimal score possible
        max_score: float, maximal score possible
        target_func: function handler, there is an assumption that the function is monotonical vs. score
        target_func_val: float, expected value of the function, such that
             target_func_val @ output_score = target_func_val
        target_func_val_precision: int, since the search is done over float outputs of target_func_val
            there is a need to define a precision at which we compare the results
    Returns:
        float, output_score such that target_func_val @ output_score = target_func_val 
        Note that if target_func_val is not reachable the output_score will match the closes function value
    """
    min_s = min_score
    max_s = max_score
    min_s_func_val = target_func(min_score=min_s)
    mid_s = min_s + (max_s - min_s) / 2
    mid_s_func_val = round(target_func(min_score=mid_s), target_func_val_precision)
    if mid_s_func_val <= min_s_func_val:
        move_min_to_mid = lambda mid, target: mid > target
    else:
        move_min_to_mid = lambda mid, target: mid < target
    while max_s - min_s > SCORE_PRECISION_:
        if mid_s_func_val == target_func_val:
            return mid_s
        elif move_min_to_mid(mid_s_func_val, target_func_val):
            min_s = mid_s
        else:
            max_s = mid_s
        mid_s = min_s + (max_s - min_s) / 2
        mid_s_func_val = round(target_func(min_score=mid_s), target_func_val_precision)
    return max_s
    
def get_working_point_based_on_metrics(df_matches, target_metrics, target_value, target_value_precision,
                                       min_det_score, max_det_score, max_range):
    """Determines the score to threshold detections.
    Parameters:
        df_matches: pd.DataFrame, data frame with matches between intruders and detections
        target_metrics: str, what metrics to use to determine the score
        target_value: float, expected value of the metrics
        target_value_precision: int, with which precision to calculate the value 
        min_det_score: float, minimum detection score
        max_det_score: float, maximum detection score
    Returns:
        float, the score to threshold the detections
    """
    thresh_score = None 
    search_score_func = None 
    if 'far' in target_metrics:
        search_score_func = partial(compute_false_alarms_per_hour, df_matches)
    elif 'fppi' in target_metrics:
        search_score_func = partial(compute_false_positives_per_image, df_matches)
    elif 'fl_dr' in target_metrics:
        search_score_func = partial(compute_probability_of_detection_of_planned_intruders, 
                                    df_matches, max_range=max_range, min_range=0)

    if search_score_func is not None:
        log.info('%s = %.5f will be used as metrics for score threshold search', target_metrics, 
                                                                                    target_value)
        thresh_score = search_score_for_target_func(min_det_score, max_det_score, 
                                        search_score_func, target_value, target_value_precision)
    return thresh_score

def get_max_range_based_on_encounters_info(df_encounters):
    """looks at all the frames with valid encounters and returns their maximum range"""
    assert is_in_df_columns(df_encounters, ['range_distance_m']), ('range_distance_m not in data frame, '
                                                                    'cannot calculate maximum range')
    return get_valid_encounters(df_encounters)['range_distance_m'].max()

def _is_min_score_as_expected(df_matches, expected_min_score):
    """assert that minimum score in the results is as expected"""
    assert is_in_df_columns(df_matches, ['s']), ('s not in data frame, cannot check minimum score')
    min_score_results = df_matches['s'].min()
    assert min_score_results >= expected_min_score, ('Expected min score = {} is greater than '
                'minimal score = {} in the results'.format(expected_min_score, min_score_results))
    return min_score_results

def _summarize_encounter_detection_rate(summary, eval_criteria, num_det_encs, num_total_encs):
    summary[eval_criteria] = OrderedDict()
    summary[eval_criteria]['Encounters'] = OrderedDict()
    for max_range in RANGES_TO_DETECT:
        summary[eval_criteria]['Encounters'][max_range] = OrderedDict()
        for num_key, num_value in num_det_encs[max_range].items():
            if num_total_encs[num_key]:
                dr_enc = float(num_value / num_total_encs[num_key])
            else:
                dr_enc  = 0.0
            summary[eval_criteria]['Encounters'][max_range][num_key] = {
                                                        'detected': int(num_value), 
                                                        'total': int(num_total_encs[num_key]),
                                                        'dr': dr_enc}
            log.info('Max. range %d: %s: %d / %d  = %.3f', max_range, num_key, num_value, 
                                                            num_total_encs[num_key], dr_enc)
    return summary

####################################################################################################
# MAIN
####################################################################################################

def run(flags):
    log.info('Reading ground truth detection matches from %s', 
                                    flags.groundtruth_results_matches_filename)
    df_gt_det_matches = pd.read_csv(flags.groundtruth_results_matches_filename, low_memory=False)
    
    min_score_results = _is_min_score_as_expected(df_gt_det_matches, flags.min_det_score)
    flags.min_det_score = max(min_score_results, flags.min_det_score)

    log.warning('Reading ground truth with encounters from %s', 
                                    flags.encounters_with_groundtruth_filename)
    df_encounters = pd.read_csv(flags.encounters_with_groundtruth_filename, low_memory=False)
    df_encounters = exclude_encounters_from_evaluation(df_encounters, flags.min_enc_range_upper_bound,
                                                      flags.max_enc_range_lower_bound)
    max_encounter_range = get_max_range_based_on_encounters_info(df_encounters)
    log.info('Maximum range of encounter is %.2f', round(max_encounter_range, 2))
    
    if flags.target_metrics is not None:    
        log.info('Determining threshold for detection score')
        if flags.target_value is None:
            if flags.target_metrics != 'far':
                raise ValueError('Please provide target value for {}'.format(flags.target_metrics))
            target_value = TARGET_FAR_PER_HOUR
        else:
            target_value = flags.target_value 
        log.info('Will use {} target value for {} calculation'.format(target_value, flags.target_metrics))
        thresh_score = get_working_point_based_on_metrics(df_gt_det_matches, flags.target_metrics, 
            target_value, flags.target_value_precision, flags.min_det_score, flags.max_det_score, 
                                                                            max_encounter_range)
    else:
        log.info('The provided minimum detection score %.5f will be used', flags.detection_score_threshold)           
        thresh_score = max(flags.min_det_score, flags.detection_score_threshold)
        
    log.info('Frame level metrics calculation for score threshold = {}'.format(thresh_score))
    df_no_dupl_objs = df_gt_det_matches.drop_duplicates(['img_name', 'id'])
    num_planned = len(df_no_dupl_objs.query(PLANNED_INTRUDERS))
    num_non_planned = len(df_no_dupl_objs.query(NON_PLANNED_INTRUDERS))
    num_non_planned_aircraft = len(df_no_dupl_objs.query(NON_PLANNED_AIRCRAFT))
    far = compute_false_alarms_per_hour(df_gt_det_matches, min_score=thresh_score) 
    fppi = compute_false_positives_per_image(df_gt_det_matches, min_score=thresh_score)
    fl_dr_range, num_det_range, num_tot_range = compute_probability_of_detection_of_planned_intruders(
                df_gt_det_matches, min_score=thresh_score, max_range=max_encounter_range, min_range=0)
    fl_dr_above_area, num_det_above_area, num_tot_above_area = compute_probability_of_detection_small_objects(
                        df_gt_det_matches, min_area=flags.min_object_area, min_score=thresh_score, 
                        obj_query= GT_ABOVE_AREA_QUERY + ' and ' + NOT_BIRD_QUERY)
    fl_dr_below_area, num_det_below_area, num_tot_below_area = compute_probability_of_detection_small_objects(
                        df_gt_det_matches, min_area=flags.min_object_area, min_score=thresh_score,
                        obj_query= GT_BELOW_AREA_QUERY + ' and ' + NOT_BIRD_QUERY)

    df_val_encounters = get_valid_encounters(df_encounters)
    df_val_encounters_with_fl_dr  = compute_moving_frame_level_detection_rate_per_encounter(df_gt_det_matches, 
        df_val_encounters, thresh_score, flags.fl_dr_temporal_win, use_track_fl_dr=flags.use_track_fl_dr)
    df_final_results = compute_encounter_detections(df_val_encounters_with_fl_dr, 
                    flags.min_fl_dr, flags.fl_dr_temporal_win, use_track_fl_dr=False)
    if flags.use_track_fl_dr:
        df_final_results_track_fl_dr = compute_encounter_detections(df_val_encounters_with_fl_dr, 
                    flags.min_fl_dr, flags.fl_dr_temporal_win, use_track_fl_dr=True)

    # saving intermidiate results
    log.info('Saving results')
    if flags.output_dir_path is None:
        output_dir = os.path.dirname(flags.groundtruth_results_matches_filename)
    else:
        output_dir = flags.output_dir_path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
   
    working_point_far_str = str(round(far, DEFAULT_METRICS_VALUE_PRECISION_)).replace('.', '_')
    if flags.save_intermediate_results:
        log.info('Saving intermediate results')
        results_filename = os.path.join(output_dir, 
            flags.results_name_prefix + '_moving_{}_fl_dr_far_{}.csv'.format(
                                                flags.fl_dr_temporal_win, working_point_far_str))
        log.info('Data frame with moving frame level detection rate is saved to %s', results_filename)
        df_val_encounters_with_fl_dr.to_csv(results_filename)

    results_filename = os.path.join(output_dir, 
        flags.results_name_prefix + '_moving_{}_fl_dr_{}_encounter_detections_far_{}'.format(
            flags.fl_dr_temporal_win, str(flags.min_fl_dr).replace('.','p'), working_point_far_str))
    df_final_results.to_csv(results_filename + '.csv')
    df_final_results_track_fl_dr.to_csv(results_filename + '_tracking.csv')
    log.info('Data frame with information on encounter detection is saved to %s.csv and %s_tracking.csv',
                                                                results_filename, results_filename)
    df_final_results.to_json(results_filename + '.json', orient='records', lines=True, indent=4)
    df_final_results_track_fl_dr.to_csv(results_filename + '_tracking.csv')

    log.info('Data frame with information on encounter detection is saved to %s.json', results_filename)
    
    # calculation of metrics
    log.info('Calculating final summary')
    num_total_encounters = {}
    for query_key, query_value in HORIZON_QUERIES.items():
        num_total_encounters[query_key] = len(df_final_results.query(query_value))
    num_det_encounters = {}
    if flags.use_track_fl_dr:
        num_det_encounters_tracking = {}
    for det_range in RANGES_TO_DETECT: 
        num_det_encounters[det_range] = {}
        num_det_encounters_tracking[det_range] = {}
        for query_key, query_value in HORIZON_QUERIES.items():
            num_det_encounters[det_range][query_key] = df_final_results.query(query_value)[
                                                        'detected_before_{}'.format(det_range)].sum()
            if flags.use_track_fl_dr:
                num_det_encounters_tracking[det_range][query_key] = df_final_results_track_fl_dr.query(
                                            query_value)['detected_before_{}'.format(det_range)].sum()

    summary = {} 
    log.info('Summary')
    summary['gt_encounters'] = flags.encounters_with_groundtruth_filename
    summary['gt_det_matches'] = flags.groundtruth_results_matches_filename
    summary['target_metrics'] = flags.target_metrics
    summary['target_value'] = flags.target_value
    summary['min_det_score'] = float(thresh_score)
    log.info('The minimum detection score is %.3f', thresh_score)
    summary['fppi'] = float(fppi)
    log.info('FPPI: %.5f', fppi)
    summary['far'] = float(far)
    log.info('HFAR: %.5f', far)
    summary['num_planned_intruders'] = int(num_planned)
    summary['num_non_planned_intruders'] = int(num_non_planned)
    summary['num_non_planned_aircraft'] = int(num_non_planned_aircraft)
    log.info('Planned Aircraft: %d', num_planned)
    log.info('Non-Planned Airborne: %d', num_non_planned)
    log.info('Non-Planned Aircraft: %d', num_non_planned_aircraft)
    tot_aircraft = num_non_planned_aircraft + num_planned
    log.info('All Aircraft: %d', tot_aircraft)
    summary['max_range'] = float(max_encounter_range)
    summary['tot_aircraft_in_range'] = int(num_tot_range)
    summary['det_aircraft_in_range'] = int(num_det_range)
    summary['fl_dr_in_range'] = float(fl_dr_range)
    log.info('AFDR, aircraft with range <= %.2f: %.5f = %d / %d', 
                                     max_encounter_range,fl_dr_range, num_det_range, num_tot_range)
    tot_aircraft_included_in_fl_dr_area = num_tot_above_area + num_tot_below_area
    assert tot_aircraft == tot_aircraft_included_in_fl_dr_area, (
     'Expected number of aircraft is {}, but got {} '.format(tot_aircraft, 
                                                            tot_aircraft_included_in_fl_dr_area))
    summary['thresh_area'] = float(flags.min_object_area)
    summary['tot_aircraft_above_area'] = int(num_det_above_area)
    summary['det_aircraft_above_area'] = int(num_tot_above_area)
    summary['fl_dr_above_area'] = float(fl_dr_above_area)
    log.info('AFDR, aircraft with area > %d: %.5f = %d / %d', 
                    flags.min_object_area, fl_dr_above_area, num_det_above_area, num_tot_above_area)
    summary['tot_aircraft_below_area'] = int(num_det_below_area)
    summary['det_aircraft_below_area'] = int(num_tot_below_area)
    summary['fl_dr_below_area'] = float(fl_dr_below_area)
    log.info('AFDR, aircraft with area <= %d: %.5f = %d / %d', 
                    flags.min_object_area, fl_dr_below_area, num_det_below_area, num_tot_below_area)
    log.info('Detected Encounters based on Detections: ')
    summary = _summarize_encounter_detection_rate(summary, 'Detection', num_det_encounters, 
                                                                        num_total_encounters)
    if flags.use_track_fl_dr:
        log.info('Detected Encounters based on Tracking: ')
        summary = _summarize_encounter_detection_rate(summary, 'Tracking', num_det_encounters_tracking, 
                                                                            num_total_encounters)
    summary_json = os.path.join(output_dir, 'summary_far_{}_min_intruder_fl_dr_{}_in_win_{}.json'.format(
            working_point_far_str, str(flags.min_fl_dr).replace('.','p'), flags.fl_dr_temporal_win))
    log.info('Saving summary to %s', summary_json)
    with open(summary_json, 'w') as fj:
        json.dump(summary, fj, indent=4)

    return far, summary_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates airborne metrics given encounters')
    add_flags(parser)
    parser.add_argument('--log-level', default=logging.getLevelName(logging.INFO), 
                                                    help='Logging verbosity level')
    args = parser.parse_args()
    setup_logging(args.log_level)    
    check_flags(args)
    run(args)

