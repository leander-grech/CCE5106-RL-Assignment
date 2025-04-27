import random
import math


class Flight:
    """
       A class to represent a flight with attributes such as heading, velocity, altitude, position, and time.

       Attributes:
       ----------
       TUPLE5KEYS : tuple
           Keys for the flight data tuple.

       Methods:
       -------
       __init__(self, heading=None, velocity=None, altitude=None, x=0., y=0., time=0):
           Initializes a Flight object with optional parameters for heading, velocity, altitude, x, y, and time.

       apply_fl_clearance(self, t, delta_alt):
           Applies a flight level clearance at a specified time.

       change_altitude(self, delta):
           Changes the altitude by a specified delta.

       change_velocity(self, delta):
           Changes the velocity by a specified delta.

       advance_clock(self, delta_time):
           Advances the flight's clock by a specified delta time.

       get_heading_unit_vectors(self):
           Returns the unit vectors of the flight's heading.

       reverse_clock(self, delta_time, allow_neg_time=False):
           Reverses the flight's clock by a specified delta time.

       get_5_tuple(self):
           Returns the flight's data as a tuple.

       get_5_tuple_dict(self):
           Returns the flight's data as a dictionary.

       __repr__(self):
           Returns a string representation of the Flight object.
   """
    TUPLE5KEYS = ('x', 'y', 'altitude', 'velocity', 'heading')

    def __init__(self, heading=None, velocity=None, altitude=None, x=0., y=0., time=0, VEL_MIN=100, ALT_MIN=100):
        self.VEL_MIN = VEL_MIN
        self.ALT_MIN = ALT_MIN
        self.heading = heading if heading is not None else random.uniform(0, 360)  # Random heading between 0 and 360 degrees
        self.velocity = velocity if velocity is not None else random.uniform(100, 1000)  # Random velocity between 100 and 1000 units
        self.altitude = altitude if altitude is not None else random.uniform(1000, 10000)  # Random altitude between 1000 and 10000 units
        self.x = x  # Initial x position
        self.y = y  # Initial y position
        self.time = time  # Initial time
        self.cur_idx = 0
        self.evo_history = {time: self.get_5_tuple()}
        self.vel_change_history = {}
        self.alt_change_history = {}

    def change_altitude(self, delta):
        self.altitude += delta
        if self.altitude < self.ALT_MIN:
            self.altitude = self.ALT_MIN  # Prevent negative altitude
        self.alt_change_history[self.time] = self.altitude

    def change_velocity(self, delta):
        self.velocity += delta
        if self.velocity < self.VEL_MIN:
            self.velocity = self.VEL_MIN  # Prevent negative velocity
        self.vel_change_history[self.time] = self.velocity

    def advance_clock(self, delta_time):
        self.time += delta_time
        self.altitude = self.alt_change_history.get(self.time, self.altitude)
        self.velocity = self.vel_change_history.get(self.time, self.velocity)

        distance = self.velocity * delta_time
        ux, uy = self.get_heading_unit_vectors()
        self.x += distance * ux
        self.y += distance * uy
        self.cur_idx += 1
        self.evo_history[self.time] = self.get_5_tuple()

    def get_heading_unit_vectors(self):
        rad_heading = math.radians(self.heading)
        dx = math.cos(rad_heading)
        dy = math.sin(rad_heading)

        return dx, dy

    def reverse_clock(self, delta_time, allow_neg_time=False):
        self.time -= delta_time
        if self.time < 0 and not allow_neg_time:
            self.time = 0  # Prevent negative time

        self.cur_idx -= 1
        if self.cur_idx < 0:
            self.cur_idx = 0
        distance = self.velocity * delta_time
        rad_heading = math.radians(self.heading)
        self.x -= distance * math.cos(rad_heading)
        self.y -= distance * math.sin(rad_heading)
        self.evo_history[self.time] = self.get_5_tuple()

    def get_5_tuple(self):
        return self.x, self.y, self.altitude, self.velocity, self.heading

    def get_5_tuple_dict(self):
        flight_data = {}
        for k, v in (self.TUPLE5KEYS, self.get_5_tuple()):
            flight_data[k] = v
        return flight_data

    def __repr__(self):
        return (f"Flight (heading={self.heading:.2f}, velocity={self.velocity:.2f}, "
                f"altitude={self.altitude:.2f}, x={self.x:.2f}, y={self.y:.2f}, time={self.time:.2f})")
