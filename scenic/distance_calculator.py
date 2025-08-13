import math
from shapely.geometry import Polygon


class VehicleDistanceCalculator:
    VEHICLE_DIMENSIONS = {
        "ego": (4.69, 1.85),
        "adversary": (4.18, 1.83)
    }

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def get_collision_box(position, heading, length, width):
        x, y = position
        half_len = length / 2
        half_wid = width / 2
        dx_forward = math.cos(heading)
        dy_forward = math.sin(heading)

        dx_left = -math.sin(heading)
        dy_left = math.cos(heading)

        vertices = [
            (x + dx_forward * half_len + dx_left * half_wid,
             y + dy_forward * half_len + dy_left * half_wid),
            (x + dx_forward * half_len - dx_left * half_wid,
             y + dy_forward * half_len - dy_left * half_wid),
            (x - dx_forward * half_len - dx_left * half_wid,
             y - dy_forward * half_len - dy_left * half_wid),
            (x - dx_forward * half_len + dx_left * half_wid,
             y - dy_forward * half_len + dy_left * half_wid)
        ]
        return vertices

    @classmethod
    def calculate_min_distance(cls, ego_pos, ego_heading, adv_pos, adv_heading):
        ego_len, ego_wid = cls.VEHICLE_DIMENSIONS["ego"]
        adv_len, adv_wid = cls.VEHICLE_DIMENSIONS["adversary"]

        ego_box = cls.get_collision_box(ego_pos, ego_heading, ego_len, ego_wid)
        adv_box = cls.get_collision_box(adv_pos, adv_heading, adv_len, adv_wid)

        ego_poly = Polygon(ego_box)
        adv_poly = Polygon(adv_box)

        if not ego_poly.is_valid:
            ego_poly = ego_poly.buffer(0)
        if not adv_poly.is_valid:
            adv_poly = adv_poly.buffer(0)

        return ego_poly.distance(adv_poly)