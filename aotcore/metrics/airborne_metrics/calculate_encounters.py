# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
""" Encounter calculation script
INPUT: ground truth json or csv (preferred) file. 
IMPLEMENTED ALGORITHM:
The algorithm for calculating the encounters based on data frame representation: 
1. Sort all the frames by 'flight_id', 'id', 'time', 'frame'
2. Find difference between consecutive frames
3. Find change in flights (if the data frame contains more than one flight)
4. Label all frames with encounter id, switch encounter id if:
    4.1 Frame difference is above maximum gap premitted
    OR
    4.2 Flight changed (not 4.1 might cover this as well, but we would like to avoid edge cases, in which
    by chance one flight ends with frame = n adn the next flight starts frame = n+1)
5. Find length of encounters and filter those that are below minimum valid length, those are the valid encounters
Note: in this approach the encounter implementation we keep things by allowing gaps at the beginning of encounter.
We allow only small gaps of maximum 3 consecutive frames, so that should not be an issue.

OUTPUT: 
Data frame (saved as .csv) with original ground truth information and added encounter information. 
Data frame (saved as .csv) with only VALID encounters' information (Valid encounters have pre-defined length)
JSON file with only valid encounters' information  - records of encounters. 

Encounter information includes:
encounter ID, flight ID, list of image names that correspond to this encounter,
minimum (first) frame, maximum (last) frame,  count of valid frames, length of the encounter with gaps
"""

import argparse
from collections import OrderedDict
from functools import partial
import json
import logging
import numpy as np
import os
import pandas as pd

from .pandas_utils import is_in_df_columns, get_deeplearning_groundtruth_as_data_frame 
from .script_utils import setup_logging, assert_file_format
#############################################
# Defaults
###############################################
DEFAULT_MAX_GAP_ = 3
DEFAULT_MIN_LENGTH_ = 30
DEFAULT_MAX_RANGE_M = 700 # encounters will only contain intruders with maximal range of 700 m.
DEFAULT_MIN_DETECTION_RANGE_M = 300 
ENC_END_RANGE_SCALE = 1.1

RANGE_INFO_FUNCS_ = ['min', 'max', 'median', 'mean'] # these functions will be used to describe 
                                                      # encounters based on the intruder's range
ENC_RANGE_DESCRIPTORS = ['{}_enc_range'.format(func) for func in RANGE_INFO_FUNCS_]
ENC_FRAME_DESCRIPTORS = ['flight_id', 'framemin', 'framemax', 'framecount', 'enc_len_with_gaps']
ENC_OTHER_DESCRIPTORS = ['is_above_horizon']    
ENC_DESCRIPTORS = ENC_RANGE_DESCRIPTORS + ENC_FRAME_DESCRIPTORS + ENC_OTHER_DESCRIPTORS     
ENC_RANGE_INFO_REDUCER = 'first'
ENC_FRAME_INFO_REDUCER = 'first'
ENC_OTHER_INFO_REDUCER = np.mean                       
##############################################
# Script related code
##############################################
log = logging.getLogger(__name__)

def add_flags(parser):
    """Utility function adding command line arguments to the parser"""
    parser.add_argument('--deeplearning-groundtruth', '-g', required=True, 
                        help='Path to the ground truth file '
                        '(consider providing the groundtruth in .csv format)')
    parser.add_argument('--output-dir-path', '-o', required=True,
                        help='Desired path for the output data frame with encounters')
    parser.add_argument('--max-range', '-r', type=int, default=DEFAULT_MAX_RANGE_M, 
                        help='Maximum range of an intruder in valid encounter')
    parser.add_argument('--min-valid-encounter-length', type=int, default=DEFAULT_MIN_LENGTH_,
                        help='Minimum number of frames with valid ground truth in valid encounter '
                        '(default: 30)')
    parser.add_argument('--max-gap-allowed', type=int, default=DEFAULT_MAX_GAP_,
                        help='Maximum size of a gap in frames with valid ground truth allowed '
                        'in a valid encounter (default: 3)')
    parser.add_argument('--min-enc-range-upper-bound', type=float,  
                        default=ENC_END_RANGE_SCALE * DEFAULT_MIN_DETECTION_RANGE_M,
                        help='The minimum range of the encounter should be not less than this value')
  
def check_flags(flags):
    """Utility function to check the input"""
    log.info('Asserting %s format', flags.deeplearning_groundtruth)
    assert_file_format(flags.deeplearning_groundtruth)
    assert not os.path.isfile(flags.output_dir_path), ('Directory name is expected as output path, '
                                                      'received {}'.format(flags.output_dir_path))
##################################################
# Encounter finder + information calculation code 
##################################################
def augment_with_frame_difference(df):
    """add a column to the data frame df with frame differences
    this function assumes that df is sorted with respect to frames"""
    assert is_in_df_columns(df, ['frame']), (
                'frame column is missing, cannot augment with frame difference')
    frame_diff = df['frame'].diff().fillna(0)
    df = df.assign(frame_diff = frame_diff) 
    return df 

def augment_with_flight_change(df):
    """add a column to the data frame df with 1 one flight changes and 0 otherwise
    this function assumes that df is sorted with respect to flights""" 
    assert is_in_df_columns(df, ['flight_id']), (
                'flight_id column is missing, cannot augment with flight change')
    # enumerate different flights
    df = df.assign(flights_code = pd.Categorical(df['flight_id']).codes)
    # calculate differences between subsequent 
    # for the first flight we will get NaN, which is set to 0 with fillna 
    flight_diff = df['flights_code'].diff().fillna(0) 
    df = df.assign(flight_changed = flight_diff)
    return df

def augment_with_encounter_switch(max_gap):
    def augment_with_encounter_switch_with_max_gap(df):
        """add a column to the data frame df with 1 when encounter index 
        should be switched and 0 otherwise
        """
        assert is_in_df_columns(df, ['frame_diff', 'flight_changed']), (
          'frame_diff and  flight_changed columns are missing, cannot augment with encounter switch')
        switch_encounter_index = [(frame_diff < 0) or (frame_diff > max_gap) or (flight_changed == 1) 
                      for frame_diff, flight_changed in zip(df['frame_diff'], df['flight_changed'])]
        df = df.assign(switch_enc_index = switch_encounter_index) 
        return df
    return augment_with_encounter_switch_with_max_gap

def augment_encounters_with_frame_info(df_encounters):
    """add 4 columns to a dataframe of encounters with: 1)the first frame, 2) the last frame,
    3) the count of frames with valid ground truth, 4) the total length/ duration of encounters with 
    gaps 
    """
    assert is_in_df_columns(df_encounters, ['encounter_id', 'frame']), ('encounter_id and frame '
                  'columns are missing, cannot augment with encounter minimum and maximum frame')
    # Next we group all the rows that correspond to the same encounter and calculate the minimum 
    # and maximum frames
    # those are the first and last frames of aec encounter
    # we also calculate the length in frames - only valid frames and all frames
    df_agg_by_enc_len = df_encounters[['encounter_id', 'frame']].groupby('encounter_id').agg(
                                                  {'frame': ['min', 'max' ,'count']}).reset_index(0)
    # the data frame will have hirerchical headers, so we concatenate for convenience. 
    df_agg_by_enc_len.columns = list(map(''.join, df_agg_by_enc_len.columns.values))
    # we also calculate the actual length of the encounter with gaps (that is including frames with 
    # missing ground truth)
    df_agg_by_enc_len = df_agg_by_enc_len.assign(enc_len_with_gaps = 
                                  df_agg_by_enc_len['framemax'] - df_agg_by_enc_len['framemin'] + 1)
    df_encounters = df_encounters.merge(df_agg_by_enc_len, on=['encounter_id'], how='left')
    return df_encounters

def augment_encounters_with_range_info(df_encounters):
    """add aggregated range per encounter, where aggregation is done
    based on the provided aggregation_func_name"""
    assert is_in_df_columns(df_encounters, ['encounter_id', 'range_distance_m']), ('encounter_id '
        'and frame columns are missing, cannot augment with encounter minimum and maximum frame')
    df_agg_by_enc_range = df_encounters[['encounter_id','range_distance_m']].groupby(
                'encounter_id').agg({'range_distance_m': RANGE_INFO_FUNCS_}).reset_index(0)
    # arrange proper name for the column with aggregated range
    df_agg_by_enc_range.columns = list(map(''.join, df_agg_by_enc_range.columns.values))
    for func in RANGE_INFO_FUNCS_:
      df_agg_by_enc_range = df_agg_by_enc_range.rename(columns={
          'range_distance_m{}'.format(func): '{}_enc_range'.format(func)})
    df_encounters = df_encounters.merge(df_agg_by_enc_range, on=['encounter_id'], how='outer')
    return df_encounters

def get_valid_encounters_info(df_encounters):
    """This function returns information for valid encounters including:
    df_encounter_info: DataFrame, encounter information with respect to descriptors
    df_encounter_images: DataFrame, all the images that below to specific encounter
    df_encounter_stats: DataFrame, statistics of the encounter descriptors 
    """
    assert is_in_df_columns(df_encounters, ['is_valid_encounter']), (
        'is_valid_encounter column is missing, cannot provide valid encounter information')
    # filter only valid encounters 
    df_valid_encounters = df_encounters.query('is_valid_encounter == True')
    # group by encounter id and calculate for each encounter its decriptors.
    agg_funcs = OrderedDict() 
    for frame_col in ENC_FRAME_DESCRIPTORS:
      agg_funcs.update({frame_col: ENC_FRAME_INFO_REDUCER})
    for frame_col in ENC_OTHER_DESCRIPTORS:
      agg_funcs.update({frame_col: ENC_OTHER_INFO_REDUCER})
    for range_col in ENC_RANGE_DESCRIPTORS:
      agg_funcs.update({range_col: ENC_RANGE_INFO_REDUCER})
    df_encounter_info = df_valid_encounters.groupby(['encounter_id'])[ENC_DESCRIPTORS].agg(
                                                                          agg_funcs).reset_index(0)
    df_encounter_images = df_valid_encounters.groupby(['encounter_id'])[['img_name']].agg(
                                                                  lambda x: list(x)).reset_index(0)
    df_encounter_stats = df_encounter_info[ENC_FRAME_DESCRIPTORS + ENC_RANGE_DESCRIPTORS].describe()
    return df_encounter_info, df_encounter_images, df_encounter_stats

def augment_with_encounters(df_planned_intruders, min_valid_encounter_length, 
                            max_gap_allowed, encounters_augmentations):
    """add encounters' information to a data frame df_intruders 
    Note: assumption is that df_planned_intruders contains only frames with planned intruders in the 
    relevant range
    """    
    # preprocessing to figure out when encounters happen 
    df_intruders_sorted = df_planned_intruders.sort_values(by=
                                                  ['flight_id', 'id','time', 'frame'])
    pre_processing_augmentations=[augment_with_frame_difference, augment_with_flight_change, 
                                  augment_with_encounter_switch(max_gap_allowed)]
    for augmentation in pre_processing_augmentations:
        df_intruders_sorted = augmentation(df_intruders_sorted)
    
    # rolling over each row in the data frame and enumerating encounters
    def enumerate_encounters_starting_from(ind_to_assign):
        index_to_assign = ind_to_assign
        def label_encounter(x):    
            nonlocal index_to_assign
            index_to_assign += (1 if x.values[0] else 0)
            return index_to_assign
        return label_encounter

    encounter_ids = df_intruders_sorted['switch_enc_index'].rolling(window=1).apply(
                                                            enumerate_encounters_starting_from(0))
    df_intruders_with_encounters = df_intruders_sorted.assign(encounter_id = encounter_ids)
    
    # adding additional information to encounters like length of the encounters, first and last frame etc.
    for augmentation in encounters_augmentations:
        df_intruders_with_encounters = augmentation(df_intruders_with_encounters)

    # valid encounter are encounters with number of frames with gt grater than minimum pre-defined length
    df_intruders_with_encounters = df_intruders_with_encounters.assign(
      is_valid_encounter = df_intruders_with_encounters['framecount'] >= min_valid_encounter_length)
    
    # final sorting
    df_intruders_with_encounters = df_intruders_with_encounters.sort_values(
                                by=['flight_id', 'encounter_id','time', 'frame']).reset_index(0)
    
    return df_intruders_with_encounters

def exclude_encounters_from_evaluation(df_encounters, min_enc_range_upper_bound, 
                                                      max_enc_range_lower_bound = None):
    """This function excludes encounters from evaluation based on provided range for detection. 
    Those encounters do not comply with the conditions below are set as invalid
    The encounter is considered invalid for evaluation given specific range_for_detection if :
    1) Encounter does not get close enough to the camera: 
    min_enc_range > min_enc_range_upper_bound 
    2) Encounter does not start at range that allows detection. The assumption here is that 
    an airborne alert requires a detection within a temporal segment of 3 secs 
    and given intruders with velocity = 60 m/s that will require the intruder to appear at least 180m 
    before the range at which the detection is required.
    max_enc_range < max_enc_range_lower_bound (by default we do not apply this condition)
    """ 
    assert is_in_df_columns(df_encounters, ['min_enc_range', 'max_enc_range']), ('min_enc_range or '
        'max_enc_range columns are missing, cannot augment with encounter minimum and maximum frame')
    
    min_range_too_far_away = df_encounters['min_enc_range'] > min_enc_range_upper_bound
    df_encounters.loc[min_range_too_far_away, 'is_valid_encounter'] = False
    
    if max_enc_range_lower_bound is not None:
      max_range_too_close= df_encounters['max_enc_range'] < max_enc_range_lower_bound
      df_encounters.loc[max_range_too_close, 'is_valid_encounter'] = False
    return df_encounters

####################################################################################################
# MAIN
####################################################################################################
def run(flags):    
    if not os.path.isdir(flags.output_dir_path):
      os.makedirs(flags.output_dir_path)
    # read the ground truth                         
    df_gt = get_deeplearning_groundtruth_as_data_frame(flags.deeplearning_groundtruth)
    flight_id = os.getenv('FLIGHT_ID')
    if flight_id is not None:
      df_gt = df_gt.query('flight_id == "{}"'.format(flight_id))
      groundtruth_csv = os.path.join(flags.output_dir_path, 'groundtruth.csv')
      log.info('Save groundtruth in .csv format to %s', groundtruth_csv) 
      df_gt.to_csv(groundtruth_csv)
    elif (flags.deeplearning_groundtruth.endswith('.json') 
        or flags.deeplearning_groundtruth.endswith('.json.gz')):
      log.info('Saving groundtruth in .csv format, please use .csv in the future') 
      df_gt.to_csv(flags.deeplearning_groundtruth.replace('.json', '.csv').replace('.gz', ''))

    # filter to get only ground truth with intruders in the specific range
    log.info('Filtering ground truth to get intruders in the specified range <= %2.fm.', 
                                                                                    flags.max_range)
    df_gt_in_range = df_gt.query('range_distance_m <= {}'.format(flags.max_range))

    # add encounters to the ground truth
    log.info('Finding encounters and adding their information to the ground truth')
    encounters_augmentations = [augment_encounters_with_frame_info, 
                                augment_encounters_with_range_info]    

    df_gt_with_encounters_in_range = augment_with_encounters(df_gt_in_range, 
                                        min_valid_encounter_length=flags.min_valid_encounter_length,
                                        max_gap_allowed=flags.max_gap_allowed,
                                        encounters_augmentations=encounters_augmentations)
    
    # save provided ground truth with the added encounters as data frame in .csv format
    groundtruth_encounters_df_filename = (os.path.join(flags.output_dir_path, 
          'groundtruth_with_encounters_maxRange{}_maxGap{}_minEncLen{}.csv'.format(flags.max_range,
                                          flags.max_gap_allowed, flags.min_valid_encounter_length)))
    log.info('Saving ground truth + encounters dataframe to %s', groundtruth_encounters_df_filename)
    df_gt_with_encounters_in_range.to_csv(groundtruth_encounters_df_filename) 

    df_gt_with_encounters_in_range = exclude_encounters_from_evaluation(df_gt_with_encounters_in_range, 
                                                                    flags.min_enc_range_upper_bound)
    # save encounters' information as data frame in .csv format
    df_encouters_info, df_encouters_images, df_encounter_stats = get_valid_encounters_info(
                                                                      df_gt_with_encounters_in_range)
    encounters_info_filename = os.path.join(flags.output_dir_path, 
                          'valid_encounters_maxRange{}_maxGap{}_minEncLen{}'.format(flags.max_range,
                                                                    flags.max_gap_allowed,
                                                                    flags.min_valid_encounter_length))
    log.info('Saving only valid encounters info dataframe to %s', encounters_info_filename + '.csv')
    df_encouters_info.to_csv(encounters_info_filename + '.csv') 
    df_encounter_stats.to_csv(encounters_info_filename + '_stats.csv')
    # save encounters' information in .json format
    log.info('Saving only valid encounters info in json format to %s', 
                                                                  encounters_info_filename + '.json')
    df_encouters_info  = df_encouters_info.merge(df_encouters_images, on='encounter_id', how='left')
    encounters_info_json = df_encouters_info.to_json(encounters_info_filename + '.json', 
                                                    orient='records', lines=True, indent=4)
    return groundtruth_encounters_df_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates encounters from the groundtruth')
    add_flags(parser)
    parser.add_argument('--log-level', 
                        default=logging.getLevelName(logging.INFO), help='Logging verbosity level')

    args = parser.parse_args()

    setup_logging(args.log_level)
    check_flags(args)
    run(args)
