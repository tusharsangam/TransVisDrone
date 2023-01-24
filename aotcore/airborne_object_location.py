#!/usr/bin/env python3
import math


class AirborneObjectLocation:
    class BoundingBox:
        def __init__(self, bb):
            self.left = bb[0]
            self.top = bb[1]
            self.width = bb[2]
            self.height = bb[3]

            self.bbox = bb

        def get_bbox(self):
            return self.bbox

        def get_center(self):
            return [self.left + (self.width/2), self.top + (self.height/2)]

        # def get_bbox_traditional(self):
        #     return [
        #         self.left - (self.width/2),
        #         self.top - (self.height/2),
        #         self.left + (self.width/2),
        #         self.top + (self.height/2)
        #     ]

        def get_bbox_traditional(self):
            return [
                self.left,
                self.top,
                self.left + self.width,
                self.top + self.height
            ]

        def __str__(self):
            return "BoundingBox(top=%s, left=%s, width=%s, height=%s)" \
                   % (self.top, self.left, self.width, self.height)

    def __init__(self, obj, frame, entity):
        self.object = obj
        self.frame = frame
        self.range_distance_m = None
        if 'range_distance_m' in entity['blob']:
            self.range_distance_m = entity['blob']['range_distance_m']
        self.bb = self.BoundingBox(entity['bb'])
        self.is_above_horizon = entity['labels']['is_above_horizon']

    @property
    def unplanned(self):
        return self.range_distance_m is None or self.range_distance_m != self.range_distance_m

    @property
    def planned(self):
        return not self.unplanned

    @property
    def above_horizon(self):
        return self.is_above_horizon == 1

    @property
    def below_horizon(self):
        return self.is_above_horizon == -1

    @property
    def horizon_not_clear(self):
        return self.is_above_horizon == 0

    def __str__(self):
        return "AirborneObjectLocation(object=%s planned=%s, is_above_horizon=%s, bb=%s, range_distance_m=%s)" % \
               (self.object.id, self.planned, self.is_above_horizon, self.bb, self.range_distance_m)
