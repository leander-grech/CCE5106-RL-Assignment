from typing import Any, SupportsFloat
import yaml
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from gymnasium.core import ObsType, ActType, RenderFrame
from matplotlib.patches import Rectangle
import numpy as np
import random
import math
from gymnasium import Env, spaces

from cce5106.sims.flight import Flight


class CollisionCourseSimulator:
    """
        A class to simulate the flight paths and potential collisions of multiple flights within a defined airspace.

        Attributes:
        ----------
        hotspot2airspace_factor : float
            Factor to define the hotspot limit relative to the airspace limit.
        spawn2airspace_factor : float
            Factor to define the spawn limit relative to the airspace limit.
        stagger2airspace_factor : float
            Factor to define the stagger limit relative to the airspace limit.
        DT : int
            Time step increment.
        DALT : int
            Altitude increment.
        ALT_MIN : int
            Minimum altitude.
        N_ALTS : int
            Number of allowable altitudes.
        DVEL : int
            Velocity increment.
        VEL_MIN : int
            Minimum velocity.
        N_VELS : int
            Number of allowable velocities.

        Methods:
        -------
        __init__(self, n, m, airspace_limit=10000, init_flights=True):
            Initializes a CollisionCourseSimulator object with parameters for number of flights, airspace limit, and whether to initialize flights.

        get_allowable_altitudes(self):
            Returns the allowable altitudes.

        get_allowable_velocities(self):
            Returns the allowable velocities.

        norm_pos(self, pos):
            Normalizes a position value relative to the airspace limit.

        norm_vel(self, vel):
            Normalizes a velocity value relative to the allowable velocities.

        norm_alt(self, alt):
            Normalizes an altitude value relative to the allowable altitudes.

        norm_heading(self, heading):
            Normalizes a heading value.

        get_norm_flight_data(self, idx=None, flight=None):
            Returns normalized flight data for a specified flight.

        get_norm_flight_data_prev_timestep(self, idx=None):
            Returns normalized flight data for the previous time step.

        init_flights(self):
            Initializes flights on a collision course and random flights in the airspace.

        _is_near_collision(self, x, y):
            Checks if a given flight is near the collision point.

        flight_within_limit(self, flight):
            Checks if a flight is within the airspace limit.

        reverse_to_limit(self, flights):
            Reverses the clock for flights until they are within the airspace limit.

        advance_clock(self):
            Advances the clock for all flights.

        reverse_clock(self):
            Reverses the clock for all flights.

        stagger_flights(self, flights):
            Staggers the positions of flights slightly.

        save_flight_data(self, save_path):
            Saves the flight data to a YAML file.

        from_saved_data(cls, data_yaml_fn):
            Creates a CollisionCourseSimulator object from saved data in a YAML file.

        calculate_distance(x1, y1, x2, y2):
            Calculates the distance between two points.

        is_within_altitude_range(self, alt1, alt2):
            Checks if two altitudes are within a specified range.

        find_congestion(self, flights):
            Finds areas of congestion among the flights.

        get_alt_color(self, alt):
            Returns a color representing an altitude.

        animate(self, save_path):
            Creates an animation of the flight paths and saves it to a file.

        __repr__(self):
            Returns a string representation of the CollisionCourseSimulator object.
    """

    hotspot2airspace_factor = 0.1
    spawn2airspace_factor = 0.9
    stagger2airspace_factor = 0.001

    # Time delta
    DT = 2
    # Altitude parameters
    DALT = 12   # delta FL
    ALT_MIN = 290   # FL
    N_ALTS = 10
    # Altitude separation threshold
    ALT_MIN_SEP = DALT * 2
    # Velocity parameters
    DVEL = 10   # delta m/s
    VEL_MIN = 200   # m/s
    N_VELS = 10

    def __init__(self, n, m, airspace_limit=10000, init_flights=True):
        self.n = n
        self.m = m
        self.airspace_limit = airspace_limit
        self.hotspot_limit = hotspot_limit = airspace_limit * self.hotspot2airspace_factor
        self.collision_x = random.uniform(-hotspot_limit, hotspot_limit)
        self.collision_y = random.uniform(-hotspot_limit, hotspot_limit)

        self.allowable_altitudes = self.get_allowable_altitudes()
        self.allowable_velocities = self.get_allowable_velocities()

        self.flights = []
        if init_flights:
            self.init_flights()
        self.sim_time = 0
        self.evo_history = {}

    def get_allowable_altitudes(self):
        return np.arange(self.N_ALTS) * self.DALT + self.ALT_MIN

    def get_allowable_velocities(self):
        return np.arange(self.N_VELS) * self.DVEL + self.VEL_MIN

    def norm_pos(self, pos):
        return pos / self.airspace_limit

    def norm_vel(self, vel):
        return (vel - self.VEL_MIN) / ((self.N_VELS - 1) * self.DVEL)

    def norm_alt(self, alt):
        return (alt - self.ALT_MIN) / ((self.N_ALTS - 1) * self.DALT)

    def norm_heading(self, heading):
        return heading / 360.

    def denorm_pos(self, pos):
        return pos * self.airspace_limit

    def denorm_vel(self, vel):
        return (vel * (self.N_VELS - 1) * self.DVEL) + self.VEL_MIN

    def denorm_alt(self, alt):
        return (alt * (self.N_ALTS - 1) * self.DALT) + self.ALT_MIN

    def denorm_heading(self, heading):
        return heading * 360.

    def get_flight_data(self, idx=None, flight=None, norm=True):
        assert (idx is not None and flight is None) or (idx is None and flight is not None), 'Either choose index of flight in flights or pass a flight'
        if idx is not None:
            assert idx < len(self.flights), f'Index {idx} out of range'

        if flight is None:
            flight = self.flights[idx]
        elif idx is None and flight is None:
            raise Exception('No flight selected')

        if isinstance(flight, Flight):
            x, y, alt, vel, heading = flight.get_5_tuple()
        elif isinstance(flight, list) or isinstance(flight, np.ndarray) or isinstance(flight, tuple):
            x, y, alt, vel, heading = flight
        elif isinstance(flight, dict):
            x, y, alt, vel, heading = [flight[k] for k in Flight.TUPLE5KEYS]
        else:
            raise NotImplementedError

        if norm:
            x = self.norm_pos(x)
            y = self.norm_pos(y)
            alt = self.norm_alt(alt)
            vel = self.norm_vel(vel)
            heading = self.norm_heading(heading)

        return x, y, alt, vel, heading

    def get_denorm_flight_data(self, idx=None, flight=None):
        assert (idx is not None and flight is None) or (idx is None and flight is not None), 'Either choose index of flight in flights or pass a flight'
        if idx is not None:
            assert idx < len(self.flights), f'Index {idx} out of range'

        if flight is None:
            flight = self.flights[idx]
        elif idx is None and flight is None:
            raise Exception('No flight selected')

        if isinstance(flight, Flight):
            x, y, alt, vel, heading = flight.get_5_tuple()
        elif isinstance(flight, list) or isinstance(flight, np.ndarray) or isinstance(flight, tuple):
            x, y, alt, vel, heading = flight
        elif isinstance(flight, dict):
            x, y, alt, vel, heading = [flight[k] for k in Flight.TUPLE5KEYS]
        else:
            raise NotImplementedError

        x = self.norm_pos(x)
        y = self.norm_pos(y)
        alt = self.norm_alt(alt)
        vel = self.norm_vel(vel)
        heading = self.norm_heading(heading)

        return x, y, alt, vel, heading

    def get_flight_data_prev_timestep(self, idx=None, norm=True):
        # If this is true, not enough advance_clock calls have been made
        if len(self.evo_history) <= 1:
            return self.get_flight_data(idx=idx, norm=norm)

        # Check if there is history saved for the previous time step
        prev_time = self.sim_time - self.DT
        if self.evo_history.get(prev_time, None) is None:
            raise Exception(f'Previous timestep at prev_time={prev_time} not found in self.evo_history')

        # Normalise the 5-tuple accordingly
        tuple_5 = self.flights[idx].evo_history[prev_time]
        return self.get_flight_data(flight=tuple_5, norm=norm)

    def init_flights(self):
        init_altitudes = self.allowable_altitudes.copy()
        coll_idx = np.random.choice(len(init_altitudes))
        collision_altitude = init_altitudes[coll_idx]
        init_altitudes = np.delete(init_altitudes, coll_idx)
        # Initialise flights on a collision course
        for i in range(self.n):
            heading = random.uniform(0, 360)
            velocity = np.random.choice(self.allowable_velocities)
            flight = Flight(altitude=collision_altitude, velocity=velocity, heading=heading, x=self.collision_x, y=self.collision_y, VEL_MIN=self.VEL_MIN, ALT_MIN=self.ALT_MIN)
            self.flights.append(flight)
        # Reverse only flights on collision to limit
        self.flights = self.reverse_to_limit(flights=self.flights)
        self.flights = self.stagger_flights(self.flights)

        # Initialise random flights in airspace
        spawn_limit = self.airspace_limit * self.spawn2airspace_factor
        for i in range(self.m):
            x = self.collision_x
            y = self.collision_y
            while self._is_near_collision(x, y):
                x = random.uniform(-spawn_limit, spawn_limit)
                y = random.uniform(-spawn_limit, spawn_limit)
            passive_altitude = np.random.choice(init_altitudes)
            velocity = np.random.choice(self.allowable_velocities)
            heading = random.uniform(0, 360)
            flight = Flight(altitude=passive_altitude, velocity=velocity, heading=heading, x=x, y=y, VEL_MIN=self.VEL_MIN, ALT_MIN=self.ALT_MIN)
            flight = self.reverse_to_limit([flight])[0]
            for _ in range(np.random.choice(10)): # change time to occur a bit sooner or later than t=0
                flight.reverse_clock(self.DT)
            self.flights.append(flight)

    def _is_near_collision(self, x, y):
        '''Check if given flight is near the collision point'''
        return math.sqrt((x - self.collision_x)**2 + (y - self.collision_y)**2) < self.hotspot_limit

    def flight_within_limit(self, flight):
        if max(abs(flight.x),  abs(flight.y)) < self.airspace_limit:
            return True
        else:
            return False

    def reverse_to_limit(self, flights):
        done = False
        while not done:
            done = all([not self.flight_within_limit(f) for f in flights])
            for f in flights:
                f.reverse_clock(self.DT)

        for f in flights:
            for _ in range(2):
                f.advance_clock(self.DT)    # To undo last step which puts at least one fligt OOB
        return flights

    def advance_clock(self):
        for flight in self.flights:
            flight.advance_clock(self.DT)
        self.sim_time += self.DT
        self.evo_history[self.sim_time] = [f.time for f in self.flights]

    def reverse_clock(self):
        for flight in self.flights:
            flight.reverse_clock(self.DT)
        self.sim_time -= self.DT
        self.evo_history[self.sim_time] = [f.time for f in self.flights]

    def stagger_flights(self, flights):
        stagger_limit = self.airspace_limit * self.stagger2airspace_factor
        for flight in flights:
            flight.x += random.uniform(-stagger_limit, stagger_limit)
            flight.y += random.uniform(-stagger_limit, stagger_limit)

        return flights

    def save_flight_data(self, save_path):
        sim_data = {
            'n': int(self.n),
            'm': int(self.m),
            'airspace_limit': int(self.airspace_limit),
            'collision_x': float(self.collision_x),
            'collision_y': float(self.collision_y),
            'flights': []
        }
        for flight in self.flights:
            sim_data['flights'].append({
                'heading': float(flight.heading),
                'velocity': int(flight.velocity),
                'altitude': int(flight.altitude),
                'x': float(flight.x),
                'y': float(flight.y),
                'time': int(flight.time)
            })

        with open(save_path, 'w') as f:
            yaml.dump(sim_data, f)

    @classmethod
    def from_saved_data(cls, data_yaml_fn):
        with open(data_yaml_fn, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        n = data['n']
        m = data['m']

        simulator = cls(n, m, data['airspace_limit'], init_flights=False)
        simulator.collision_x = data['collision_x']
        simulator.collision_y = data['collision_y']
        for flight_info in data['flights']:
            flight = Flight(
                heading=flight_info['heading'],
                velocity=flight_info['velocity'],
                altitude=flight_info['altitude'],
                x=flight_info['x'],
                y=flight_info['y'],
                time=flight_info['time']
            )
            simulator.flights.append(flight)
        return simulator

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return dist

    def is_within_altitude_range(self, alt1, alt2):  # 1000 feet in meters
        return abs(alt1 - alt2) < self.ALT_MIN_SEP

    def find_congestion(self, flights):
        congestion_centers = []
        congestion_altitudes = []
        xlist = [f.x for f in flights]
        ylist = [f.y for f in flights]
        altlist = [f.altitude for f in flights]
        n_flights = len(flights)
        for i in range(n_flights):
            nearby_alts = []
            nearby_alts.append(altlist[i])
            nearby_xy = []
            nearby_xy.append([xlist[i], ylist[i]])
            for j in range(n_flights - i):
                j += i
                if i != j and self.is_within_altitude_range(altlist[i], altlist[j]):
                    distance = self.calculate_distance(xlist[i], ylist[i], xlist[j], ylist[j])
                    if distance <= self.hotspot_limit:
                        nearby_alts.append(altlist[j])
                        nearby_xy.append([xlist[j], ylist[j]])
            if len(nearby_alts) >= 2:  # n - 1 others nearby means n aircraft including the current one
                # x, y = self.mpl_objs['m'](longitudes[i], latitudes[i])
                congestion_centers.append(np.mean(np.array(nearby_xy), axis=0))
                congestion_altitudes.append(np.mean(nearby_alts))
        return congestion_centers, congestion_altitudes

    def get_alt_color(self, alt):
        alt = np.clip((alt - self.ALT_MIN) / (self.DALT * len(self.allowable_altitudes)), 0, 1)
        c = mpl.cm.plasma(alt)
        return c

    def animate(self, save_path):
        fig, ax = plt.subplots()

        vmin = self.allowable_altitudes[0]
        vmax = self.allowable_altitudes[-1]
        ticks = self.allowable_altitudes
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.plasma
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # This can be an empty array or any data array if needed

        # Add a colorbar next to the axis
        cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
        cbar.set_label('Altitude')

        ax.set_xlim(-self.airspace_limit, self.airspace_limit)
        ax.set_ylim(-self.airspace_limit, self.airspace_limit)
        ax.add_patch(Rectangle((-self.hotspot_limit, -self.hotspot_limit), width=2 * self.hotspot_limit, height=2 * self.hotspot_limit, fill=False))

        arrow_size = 250
        lines = []
        fl_texts = []
        arrows = []
        for i, _ in enumerate(self.flights):
            c = self.get_alt_color(self.flights[i].altitude)
            x, y, alt, vel, heading = self.flights[i].get_5_tuple()
            arrow = mpl.patches.FancyArrow(x, y, 0, 0, edgecolor=c, facecolor=c, head_width=arrow_size, head_length=arrow_size, zorder=15)
            arrows.append(arrow)
            text = ax.text(0, 0, '', fontsize=6)
            # line, = ax.plot([], [], marker='o', label=f'Flight {i}', color=c)
            # lines.append(line)
            fl_texts.append(text)

        def init():
            nonlocal lines, fl_texts, arrows
            for i, flight in enumerate(self.flights):
                dx, dy = flight.get_heading_unit_vectors()
                arrow = mpl.patches.FancyArrow(flight.x, flight.y, dx, dy, width=0.01, color='b')
                ax.add_patch(arrow)
                arrows[i] = arrow
                # lines[i].set_data([], [])
                fl_texts[i].set_text('')
            return lines, fl_texts, arrows

        def update(frame):
            nonlocal lines, fl_texts, arrows
            for i, flight, in enumerate(self.flights):
                before_where = self.flight_within_limit(self.flights[i])
                self.flights[i].advance_clock(1)
                after_where = self.flight_within_limit(self.flights[i])
                if before_where and not after_where:
                    arrows[i].remove()
                    fl_texts[i].remove()
                    continue
                elif not before_where or not after_where:
                    continue
                x, y, alt, vel, heading = self.flights[i].get_5_tuple()
                # lines[i].set_data([x], [y])

                cong_cen, cong_alt = self.find_congestion(self.flights)
                for cen , calt in zip(cong_cen, cong_alt):
                    c_alt = self.get_alt_color(calt)
                    ax.add_patch(plt.Circle(cen, self.hotspot_limit, color=c_alt, fill=False, alpha=0.1))

                arrow_size = 250
                ux, uy = flight.get_heading_unit_vectors()
                dx = ux * arrow_size
                dy = uy * arrow_size
                c = self.get_alt_color(alt)
                arrows[i].remove()
                arrow = mpl.patches.FancyArrow(x, y, dx, dy, edgecolor=c, facecolor=c, head_width=arrow_size,
                                               head_length=arrow_size, zorder=15)
                ax.add_patch(arrow)
                arrows[i] = arrow

                fl_texts[i].set_position((x, y))
                fl_texts[i].set_text(f'FL{alt:d}\nV{vel:d}')

                ax.set_title(f'Step = {frame}\nNb. collisions = {len(cong_cen)}')
            return lines, fl_texts, arrows

        ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=False, repeat=False)

        ffwriter = mpl.animation.FFMpegWriter(fps=5)
        ani.save(save_path, dpi=100, writer=ffwriter)

    def __repr__(self):
        return (f"CollisionCourseSimulator(\n"
                f"  collision_x={self.collision_x:.2f}, collision_y={self.collision_y:.2f},\n" +
                "".join([f'flight{i}={flight}\n' for i, flight in enumerate(self.flights)]))
