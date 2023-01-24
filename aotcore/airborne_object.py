from loguru import logger


class AirborneObject:

    def __str__(self):
        return "AirborneObject#%s(num_frames=%s, planned=%s)" % (self.id, self.num_frames, self.planned)

    def __init__(self, entity):
        self.location = []
        self.id = entity['id']
        self._planned = None

    def register_location(self, obj_location):
        self.location.append(obj_location)
        if self.planned is None:
            self._planned = obj_location.planned

    @property
    def planned(self):
        return self._planned

    @property
    def num_frames(self):
        return len(self.location)
