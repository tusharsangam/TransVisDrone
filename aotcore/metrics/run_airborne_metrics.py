# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# This module binds together different modules or airborne metrics for local evaluation

import argparse
from collections import defaultdict
from glob import glob
import json
import logging
import os
import shutil

import pandas as pd 

from .airborne_metrics import calculate_airborne_metrics as calculate_airborne_metrics
from .airborne_metrics import calculate_encounters as calculate_encounters
from .airborne_metrics import match_groundtruth_results as match_groundtruth_results
from .airborne_metrics.script_utils import setup_logging, remove_extension
from .airborne_metrics.calculate_airborne_metrics import RANGES_TO_DETECT, HORIZON_QUERIES
from .airborne_metrics.calculate_encounters import DEFAULT_MAX_RANGE_M

MAX_FAR_TO_BE_RANKED = 0.5 
MAX_FPPI_TO_BE_RANKED = 0.0005
DEFAULT_MIN_TRACK_LEN_ = 0
DEFAULT_MIN_SCORE_ = 0
DEEP_LEARNING_GROUNDRUTH = 'groundtruth.json' 
ENCOUNTERS_GROUNDTRUTH = 'groundtruth_with_encounters_maxRange{max_range}_maxGap3_minEncLen30.csv'

log = logging.getLogger(__name__)

def add_flags(parser):
    """Utility function adding command line arguments to the parser"""
    parser.add_argument('--dataset-folder', '-d', required=False,
                        help='Name of the folder with dataset ground truth')
    parser.add_argument('--results-folder', '-r', required=False,
                        help='Name of the folder with results')
    parser.add_argument('--summaries-folder', '-s',
                        help='Name of the folders to hold summary files')
    parser.add_argument('--min-score', type=float, default=DEFAULT_MIN_SCORE_,
                        help='Minimum score to threshold the results if cannot be derived from the name')    
    parser.add_argument('--min-track-len', '-l', type=int, default=DEFAULT_MIN_TRACK_LEN_, 
                        help='minimum track length - the results will be evaluated for all track lengths'
                        'with at least such length')
    parser.add_argument('--baseline-far', type=float, 
                        help='Will be used if baseline-ufo-result is not provided')    
    parser.add_argument('--baseline-fppi', type=float, 
                        help='Will be used if specified')   
    parser.add_argument('--enc-max-range', '-f', default=DEFAULT_MAX_RANGE_M,
                        help='Max range of intruder in the encounters')
    
    parser.add_argument('--log-level', default=logging.getLevelName(logging.INFO), 
                        help='Logging verbosity level')
 
def call_script(script_handler, args_list):
    parser = argparse.ArgumentParser()
    script_handler.add_flags(parser)
    args = parser.parse_args(args_list)
    script_handler.check_flags(args)
    return script_handler.run(args)

def call_encounter_calculation(args_list):
    return call_script(calculate_encounters, args_list)

def call_metrics_calculation(args_list):
    return call_script(calculate_airborne_metrics, args_list)

def call_match_calculation(groundtruth, result_file, min_track_len, min_score=None):
    args_list = ['--deeplearning-groundtruth', groundtruth, 
                 '--airborne-classifier-results', result_file, 
                 '--min-track-len', str(min_track_len)]
    if min_score is not None:
        args_list += ['--detection-score-threshold', str(min_score)]
    return call_script(match_groundtruth_results, args_list)

def call_metrics_with_target_far(gt_det_match_result_path, encounters,
                                                          far=None, min_score=0):
    args_list = ['--encounters-with-groundtruth-filename', encounters, 
                 '--groundtruth-results-matches-filename', gt_det_match_result_path, 
                 '--target-metrics', 'far', 
                 '--min-det-score', str(min_score),
                 '--use-track-fl-dr']
    if far is not None:
        args_list += ['--target-value', str(far)]
    return call_metrics_calculation(args_list)

def call_metrics_with_target_fppi(gt_det_match_result_path, encounters, fppi, min_score=0):
    return call_metrics_calculation(['--encounters-with-groundtruth-filename', encounters, 
                                     '--groundtruth-results-matches-filename', gt_det_match_result_path,
                                     '--target-metrics', 'fppi', 
                                     '--min-det-score', str(min_score),
                                     '--target-value', str(fppi),
                                     '--use-track-fl-dr'])

def call_metrics_with_score(gt_det_match_result_path, encounters, score=None):
    args_list = ['--encounters-with-groundtruth-filename', encounters, 
                 '--groundtruth-results-matches-filename', gt_det_match_result_path,
                 '--use-track-fl-dr']
    if score is not None:
        args_list += ['--detection-score-threshold', str(score)]
    return call_metrics_calculation(args_list)

def calculate_airborne_encounters(groundtruth, output_dir):
    return call_script(calculate_encounters,['--deeplearning-groundtruth', groundtruth, 
                                             '--output-dir-path', output_dir])

def get_min_score_from_name(result_name):
    def replace_decimal(value):
        return str(value).replace('.', 'p')
    scores = [val / 10 for val in range(10)]
    for score in scores:
        if 't' + replace_decimal(score) in result_name:
            return score
    return None 

def _change_to_csv(filename):
    if filename.endswith('.csv'):
        return filename
    return filename.replace('.json', '.csv').replace('.gz','')

####################################################################################################
# MAIN
###################################################################################################
def summarize(summaries_dir):
    """Gathers all summaries into one table"""
    all_summaries = glob(os.path.join(summaries_dir, '*.json'))
    summaries = []
    for i, summary_f in enumerate(all_summaries):
        with open(summary_f, 'r') as jf:
            summaries.append(json.load(jf))

    for range_ in RANGES_TO_DETECT:
        results = defaultdict(list)
        for summary in summaries:
            results_name = os.path.basename(os.path.dirname(summary['gt_det_matches']))
            results['Algorithm'].append(results_name)
            results['Score'].append(summary['min_det_score'])
            results['FPPI'].append(summary['fppi'])
            results['AFDR'].append(summary['fl_dr_in_range'])
            results['Detected planned aircraft'].append(summary['det_aircraft_in_range'])
            results['Total planned aircraft'].append(summary['tot_aircraft_in_range'])
            results['HFAR'].append(summary['far'])
            for dr_criteria in ['Tracking']:
                for scenario in list(HORIZON_QUERIES.keys()):
                    results['EDR {}, {}'.format(scenario, dr_criteria)].append(
                            round(summary[dr_criteria]['Encounters'][str(range_)][scenario]['dr'], 5))
                    results['Detected {}, {}'.format(scenario, dr_criteria)].append(
                            round(summary[dr_criteria]['Encounters'][str(range_)][scenario]['detected'], 5))
                    results['Total {}, {}'.format(scenario, dr_criteria)].append(
                            round(summary[dr_criteria]['Encounters'][str(range_)][scenario]['total'], 5))
        df_to_save = pd.DataFrame.from_dict(results)
        df_to_save = df_to_save.sort_values(
            ['EDR All, Tracking', 'EDR Below Horizon, Tracking', 'EDR Mixed, Tracking', 
             'EDR Above Horizon, Tracking'], ascending=False).reset_index(drop=True).rename_axis('#')
        df_to_save.to_csv(os.path.join(summaries_dir, 'dr_encounters_detected_before_{}.csv'.format(range_)))
        df_benchmark_1 = df_to_save[['Algorithm', 'Score', 'HFAR', 'EDR All, Tracking']]
        df_benchmark_1.to_csv(os.path.join(summaries_dir, 'detection_tracking_benchmark_results_for_ranking.csv'))
        df_benchmark_2 = df_to_save[['Algorithm', 'Score', 'FPPI', 'AFDR']]
        df_benchmark_2.to_csv(os.path.join(summaries_dir, 'detection_only_benchmark_results_for_ranking.csv')) 
        

def run(flags):
    encounters_gt_path = os.path.join(flags.dataset_folder, ENCOUNTERS_GROUNDTRUTH.format(
                                                                max_range=flags.enc_max_range))
    log.info('Encounter ground truth: %s', encounters_gt_path)
    groundtruth_path = os.path.join(flags.dataset_folder, DEEP_LEARNING_GROUNDRUTH)
    groundtruth_path_csv = _change_to_csv(groundtruth_path)
    if os.path.isfile(groundtruth_path_csv):
        groundtruth_path = groundtruth_path_csv
    if flags.summaries_folder is not None:
        summaries_dir = flags.summaries_folder
    else:
        summaries_dir = os.path.join(flags.results_folder, 'summaries')
    if not os.path.isdir(summaries_dir):
        os.makedirs(summaries_dir)
    if not os.path.exists(encounters_gt_path):
        encounters_gt_path = calculate_airborne_encounters(groundtruth_path, flags.dataset_folder)
        groundtruth_path = _change_to_csv(groundtruth_path) # for acceleration of other evaluations
    
    result_files = glob(os.path.join(flags.results_folder, '*.json*')) 

    for result_file in result_files:
        result_file_base = remove_extension(result_file)
        gt_det_matches_filename = None
        min_score = get_min_score_from_name(result_file)
        if min_score is None:
            min_score = flags.min_score
        min_track_len = flags.min_track_len 
        results_dir = result_file_base + '_metrics' + '_min_track_len_{}'.format(min_track_len)
        
        if os.path.isfile(result_file_base + '.csv'):
            result_file = result_file_base + '.csv'
        # calculates matches for detections that belong to tracks with min_track_len (in online fashion) 
        # and have at least min_score 
        gt_det_matches_filename = call_match_calculation(groundtruth_path, result_file, 
                                                         min_track_len, min_score)  
        groundtruth_path = _change_to_csv(groundtruth_path) # for acceleration of other evaluations
        exp_name = os.path.basename(os.path.dirname(gt_det_matches_filename))
        # calculate metrics for all the matches with min_track_len and min_score
        
        # now perform metrics calculation for baseline far
        if flags.baseline_far:
            _, summary_json = call_metrics_with_target_far(gt_det_matches_filename, encounters_gt_path, 
                                                        min_score=min_score, far=flags.baseline_far)
        else:
            _, summary_json = call_metrics_with_score(gt_det_matches_filename, encounters_gt_path,
                                                                                score=min_score)
        # copy summary for directory of summaires
        shutil.copyfile(summary_json, os.path.join(summaries_dir, 
                os.path.basename(summary_json.replace('summary', '{}_summary'.format(exp_name)))))  

        # now perform metrics calculation for baseline fppi
        if flags.baseline_fppi:
            _, summary_json = call_metrics_with_target_fppi(gt_det_matches_filename, encounters_gt_path, 
                                                       fppi=flags.baseline_fppi, min_score=min_score)
            shutil.copyfile(summary_json, os.path.join(summaries_dir, 
                os.path.basename(summary_json.replace('summary', '{}_summary'.format(exp_name)))))
                       
    summarize(summaries_dir)

def rerun(args):
    setup_logging(args.log_level)
    run(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate airborne detection results with airborne metrics')
    add_flags(parser)
    args = parser.parse_args()
    rerun(args)
    




 
