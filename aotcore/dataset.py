#!/usr/bin/env python3
import os
import json
from loguru import logger
from .flight import Flight
from .file_handler import FileHandler


class Dataset:
    metadata = None
    flights = {}

    def __init__(self, local_path, s3_path, download_if_required=True, partial=False, prefix=None):
        self.file_handler = None
        self.partial = partial
        self.valid_encounter = {}
        self.add(local_path, s3_path, download_if_required, prefix)

    def load_gt(self):
        logger.info("Loading ground truth...")
        gt_content = self.file_handler.get_file_content(self.gt_loc)
        gt = json.loads(gt_content)

        self.metadata = gt["metadata"]
        for flight_id in gt["samples"].keys():
            flight_id_with_prefix = flight_id
            if self.prefix:
                flight_id_with_prefix = self.prefix + flight_id
            if self.partial and flight_id not in self.valid_encounter:
                #logger.info("Skipping flight, not present in valid encounters: %s" % flight_id)
                continue
            self.flights[flight_id_with_prefix] = Flight(flight_id_with_prefix, gt["samples"][flight_id], self.file_handler, self.valid_encounter.get(flight_id), prefix=self.prefix)

    def load_ve(self):
        if self.partial:
            logger.info("Loading valid encounters...")
            ve = self.file_handler.get_file_content(self.valid_encounters_loc)
            for valid_encounter in ve.split('\n\n    '):
                valid_encounter = json.loads(valid_encounter)
                if valid_encounter["flight_id"] not in self.valid_encounter:
                    self.valid_encounter[valid_encounter["flight_id"]] = []
                self.valid_encounter[valid_encounter["flight_id"]].append(valid_encounter)

    def add(self, local_path, s3_path, download_if_required=True, prefix=None):
        self.prefix = prefix
        self.file_handler = FileHandler(local_path, s3_path, download_if_required)
        self.load_ve()
        self.load_gt()

    def get_flight_ids(self):
        return list(self.flights.keys())

    @property
    def gt_loc(self):
        return 'ImageSets/groundtruth.json'

    @property
    def valid_encounters_loc(self):
        return 'ImageSets/valid_encounters_maxRange700_maxGap3_minEncLen30.json'

    def get_flight_by_id(self, flight_id):
        return self.flights[flight_id]

    def get_flight(self, flight_id):
        return self.get_flight_by_id(flight_id)

    def __str__(self):
        return "Dataset(num_flights=%s)" % (len(self.flights))


if __name__ == "__main__":
    local_path = '/Users/skbly7/Terminal/aicrowd/repos/airborne-detection-starter-kit/data'
    s3_path = 's3://airborne-obj-detection-challenge-training/part1/'
    dataset = Dataset(local_path, s3_path)
    print(dataset.flights)
