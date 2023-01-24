from loguru import logger

from .airborne_object import AirborneObject
from .airborne_object_location import AirborneObjectLocation
from .frame import Frame
import os
import subprocess


class FlightMetadata:
    """
    This holds the metadata information for all the data associated with flights.

        Example for metadata:
         {'data_path': 'train3/08455059e65d4fb7a6f4e28e49e130ac/',
         'fps': 10.0,
         'number_of_frames': 1199,
         'duration': 119900.0,
         'resolution':
             {'height': 2048, 'width': 2448}
         }
    """

    class Resolution:
        def __init__(self, height, width):
            self.height = height
            self.width = width

    def __init__(self, metadata):
        self.data_path = metadata['data_path']
        self.fps = metadata['fps']
        self.number_of_frames = metadata['number_of_frames']
        self.duration = metadata['duration']

    def __str__(self):
        return "FlightMetadata(data_path=%s, number_of_frames=%s, fps=%s)" % \
               (self.data_path, self.number_of_frames, self.fps)


class Flight:
    """
    This class holds training data related to a flight. This consists of metadata and annotated frames.
    """

    def __str__(self):
        return "Flight#%s(num_frames=%s, num_airborne_objs=%s)" % (self.id, self.num_frames, self.num_airborne_objs)

    @property
    def num_frames(self):
        return len(self.frames.keys())

    @property
    def num_airborne_objs(self):
        return len(self.detected_objects.keys())

    @property
    def location(self):
        if self.prefix:
            return 'Images/' + self.prefix + self.id
        return 'Images/' + self.id

    def __init__(self, flight_id, flight_data: dict, file_handler, valid_encounter=None, prefix=None):
        self.id = flight_id
        self.frames = {}
        self.detected_objects = {}
        self.file_handler = file_handler
        self.prefix = prefix
        self.metadata = FlightMetadata(flight_data['metadata'])
        self.valid_encounter = valid_encounter
        for entity in flight_data['entities']:
            frame_id = entity['blob']['frame']

            if self.valid_encounter is not None:
                valid = False
                for encounter in self.valid_encounter:
                    if encounter["framemin"] <= int(frame_id) <= encounter["framemax"]:
                        valid = True

                if not valid:
                    continue

            if frame_id not in self.frames:
                self.frames[frame_id] = Frame(entity, self.file_handler, self)

            if self.frame_has_airborne_object(entity):
                obj_id = entity['id']
                if obj_id not in self.detected_objects:
                    self.detected_objects[obj_id] = AirborneObject(entity)

                obj = self.detected_objects[obj_id]
                obj_location = AirborneObjectLocation(obj, self.frames[frame_id], entity)
                obj.register_location(obj_location)

                self.frames[frame_id].register_object_location(obj_location)

    @staticmethod
    def frame_has_airborne_object(entity):
        return 'id' in entity

    @property
    def flight_id(self):
        return self.id

    def get_airborne_objects(self):
        return self.detected_objects.values()

    def get_frame(self, id):
        if self.valid_encounter is not None and id not in self.frames:
            logger.info("frame_id not present in partial dataset")
            return None
        return self.frames[id]

    def get_metadata(self):
        return self.metadata

    def download(self, parallel=None):
        images = []
        for f in self.frames:
            images.append([self.frames[f].image_s3_path(), self.frames[f].image_path()])
        self.file_handler.download_from_s3_parallel(images, parallel=parallel)

    def generate_video(self, speed_x=1):
        image = self.frames[list(self.frames.keys())[0]].image_path()
        flight_folder = self.file_handler.absolute_path_to_file_locally(os.path.dirname(image))
        cur_dir = os.getcwd()
        os.chdir(flight_folder)
        if not os.path.isfile(flight_folder + 'flight.mp4'):
            logger.info("Generating video...")
            os.system(" ".join([
                'ffmpeg', '-framerate', str(10*speed_x), '-pattern_type', 'glob', '-i', "'*.png'", '-c:v', 'libx264',
                '-r', str(10*speed_x), '-pix_fmt', 'yuv420p', 'flight.mp4'
            ]))
        os.chdir(cur_dir)
        return flight_folder + '/flight.mp4'
