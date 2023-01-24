from .file_handler import FileHandler
import os
import imgaug as ia


class Frame:
    """
    This holds the information related to frame of the flight.

        Example for frame level data (multiple per frame):
        {
            'time': 1550844897919368155,
            'blob': {
                'frame': 480,
                'range_distance_m': nan
            },
            'id': 'Bird2',
            'bb': [1013.4, 515.8, 6.0, 6.0],
            'labels': {'is_above_horizon': 1},
            'flight_id': '280dc81adbb3420cab502fb88d6abf84',
            'img_name': '1550844897919368155280dc81adbb3420cab502fb88d6abf84.png'
        }
    """

    def __init__(self, entity, file_handler: FileHandler, flight):
        self.detected_object_locations = {}
        self.detected_objects = {}
        self.id = entity['blob']['frame']
        self.timestamp = entity['time']
        self.file_handler = file_handler
        self.flight = flight

    def register_object_location(self, obj_location):
        if obj_location.object.id not in self.detected_objects:
            self.detected_objects[obj_location.object.id] = obj_location.object
        self.detected_object_locations[obj_location.object.id] = obj_location

    def __str__(self):
        return "Frame#%s(time=%s, num_detected_objects=%s)" % (self.id, self.time, self.num_detected_objects)

    @property
    def time(self):
        return self.timestamp

    @property
    def frame(self):
        return self.id

    @property
    def num_detected_objects(self):
        return len(self.detected_objects)

    def image_path(self):
        flight_id = self.flight.id
        if self.flight.prefix:
            flight_id = self.flight.id.split(self.flight.prefix)[1]
        return os.path.join('Images', self.flight.id, (str(self.timestamp) + flight_id + '.png'))

    def image_s3_path(self):
        flight_id = self.flight.id
        if self.flight.prefix:
            flight_id = self.flight.id.split(self.flight.prefix)[1]
        return os.path.join('Images', flight_id, (str(self.timestamp) + flight_id + '.png'))

    def image(self, type='pil'):
        """
        Read the image of this frame in whichever type you want.

        type: It can be one of ["pil", "cv2", "numpy"]

        By default, normal PIL.Image is returned
        """
        return self.file_handler.get_file_content(s3_path=self.image_s3_path(), type=type, local_path=self.image_path())

    def image_annotated(self):
        img = self.image(type='cv2')

        obj_locations_bbox_traditional = []
        obj_locations_centers = []
        for obj_name in self.detected_object_locations:
            obj_locations_bbox_traditional.append(ia.BoundingBox(
                *self.detected_object_locations[obj_name].bb.get_bbox_traditional()))
            obj_locations_centers.append(ia.Keypoint(*self.detected_object_locations[obj_name].bb.get_center()))

        bbsoi = ia.BoundingBoxesOnImage(obj_locations_bbox_traditional, shape=img.shape)
        img_annotated = bbsoi.draw_on_image(img, size=7, color=(255, 255, 0))

        kpsoi = ia.KeypointsOnImage(obj_locations_centers, shape=img.shape)
        img_annotated = kpsoi.draw_on_image(img_annotated, size=3, color=(255, 255, 0))
        return img_annotated
