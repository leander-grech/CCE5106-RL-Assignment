from typing import Any
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gymnasium.core import ObsType, ActType, RenderFrame
from matplotlib.patches import Rectangle
import numpy as np
from gymnasium import Env, spaces

from cce5106.sims.collision_sim import CollisionCourseSimulator
from cce5106.utils.utils import *


class UMFlightEnv(Env):
    """
    A custom OpenAI Gym environment for simulating flights and potential collisions.

    Attributes:
    ----------
    MAX_CLEARANCES : int
        Maximum number of clearances to issue in one episode.

    Methods:
    -------
    __init__(self, n=2, m=0, max_steps=50):
        Initializes a DummyFlightEnv object with parameters for the number of flights and maximum steps.

    reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        Resets the environment and returns the initial observation.

    extract_observation(self):
        Returns the current state observation.

    step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        Executes one time step within the environment based on the given action.

    reward_function(self):
        Computes the reward based on the current state.

    render(self, save_path=None) -> RenderFrame | list[RenderFrame] | None:
        Renders the current state of the environment.

    __repr__(self):
        Returns a string representation of the DummyFlightEnv object.
    """
    def __init__(self, n=2, m=0, max_steps=50, rew_coeffs=None):
        self.MAX_CLEARANCES = 100           # Maximum number of clearances to issue in one episode. Beyond which, episode ends
        self.INVALID_MAXCLEARANCE_RATIO = 0.5    # E.g. if 0.5, it means that if 50% of clearances are invalid, ep terminates

        if rew_coeffs is None:
            rew_coeffs = dict(collision=1.,
                              clearance=1.,
                              invalid=1.)
        self.COLLISION_COEFF_SCALE = rew_coeffs['collision']    # Scales the collisions coefficient in the `reward_function` method
        self.CLEARANCE_COEFF_SCALE = rew_coeffs['clearance']     # Scales the clearances coefficient in the `reward_function` method
        self.INVALID_COEFF_SCALE = rew_coeffs['invalid']       # Factor multiplied to the invalid actions coefficient in reward method

        self.sim = None
        self.n, self.m = n, m
        self.n_flights = n + m

        self.cur_idx = 0
        self.max_steps = max_steps
        self.low_score = None

        '''
        ACTION SPACE
        ____________
        Flight 1:   Discrete 5 - NOOP[0], VEL_DEC[1], VEL_INC[2], FL_DEC[3], FL_INC[4]
        Flight 2:   Discrete 5 - NOOP[0], VEL_DEC[1], VEL_INC[2], FL_DEC[3], FL_INC[4]
        ...
        Flight n+m: Discrete 5 - NOOP[0], VEL_DEC[1], VEL_INC[2], FL_DEC[3], FL_INC[4]
        '''
        self.ACTIONS2IDX = {'NOOP': 0, 'VEL_DEC': 1, 'VEL_INC': 2, 'FL_DEC': 3, 'FL_INC': 4}
        self.IDX2ACTION = {v: k for k, v in self.ACTIONS2IDX.items()}

        self.n_act = len(self.ACTIONS2IDX) * self.n_flights
        self.action_space = spaces.MultiDiscrete(np.repeat(len(self.ACTIONS2IDX), self.n_flights))
        self.cnt_clearances = None
        self.clearances = None
        self.n_clearances = None
        self.n_invalids = None
        self.invalids = None

        '''
        OBSERVATION SPACE
        _________________
        Concatenated list containing flight data for the current and previous time-steps.
        '''
        self.n_obs = 5 * 2 * self.n_flights + 1 # 5-tuple x 2 time-steps x (n+m) + time
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_obs,), dtype=np.float32)
        self.current_state = None

        '''
        RENDERING
        _________
        '''
        # Define the colors and markers for each action
        self.action_colors = {
            0: 'lightgray',
            1: 'red',
            2: 'green',
            3: 'blue',
            4: 'c'
        }
        self.action_markers = {
            0: 'o',
            1: 'v',
            2: '^',
            3: '1',
            4: '2'
        }
        self.vmin = None
        self.vmax = None
        self.cmap = None
        self.norm = None

    def __str__(self):
        return f'UMFlightEnv_N{self.n}_M{self.m}_max_steps{self.max_steps}'

    def __repr__(self):
        return (f"UMFlightEnv; Simple straight-line collision simulator\n"
                f"Actions: {[item for item in self.ACTIONS2IDX]}\n"
                f"Max. nb. clearances, i.e. not NOOP = {self.MAX_CLEARANCES}\n"
                f"Nb. flights on coll. course; N={self.n}\n"
                f"Nb. flights rand. generated; M={self.m}\n"
                f"Collision coeff. scale = {self.COLLISION_COEFF_SCALE}\n"
                f"Clearance coeff. scale = {self.CLEARANCE_COEFF_SCALE}\n"
                f"Invalid coeff. scale = {self.INVALID_COEFF_SCALE}\n"
                f"Max. nb. invalid actions = {self.MAX_CLEARANCES * self.INVALID_MAXCLEARANCE_RATIO}\n"
                f"Simulator; {repr(self.sim)}")

    def reset(self, * , seed=None, options: dict[str, Any] | None = None, sim=None):
        if sim is None:
            self.sim = CollisionCourseSimulator(n=self.n, m=self.m, init_flights=True)
        else:
            self.sim = sim
        self.cur_idx = 0
        self.low_score = 0
        self.sim.advance_clock()    # So that the first two time-steps have different data
        self.n_clearances = 0
        self.n_invalids = 0
        self.cnt_clearances = []
        self.clearances = []
        self.invalids = []
        self.current_state = self.extract_observation()

        # reward shaping
        self.max_colls = 0

        # rendering
        self.vmin = self.sim.allowable_altitudes[0]
        self.vmax = self.sim.allowable_altitudes[-1]
        self.cmap = mpl.cm.plasma
        self.norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)

        return self.current_state, {}

    def extract_observation(self):
        '''
        This method returns the current state. It is dependent on the number of step calls made
        '''
        obs = []
        for idx, flight in enumerate(self.sim.flights):
            flight_data_norm = self.sim.get_flight_data(idx, norm=True)
            flight_data_prev_norm = self.sim.get_flight_data_prev_timestep(idx, norm=True)
            obs.append(flight_data_norm)            # Add current flight data to observable
            obs.append(flight_data_prev_norm)       # Add previous flight data to observable
        obs.append([self.cur_idx/self.max_steps])   # Add time to observable
        obs = np.concatenate(obs)   # Flatten
        return obs

    def step(self, action: ActType):
        self.clearances.append(action)

        min_alt, max_alt = min(self.sim.allowable_altitudes), max(self.sim.allowable_altitudes)
        min_vel, max_vel = min(self.sim.allowable_velocities), max(self.sim.allowable_velocities)

        for i, flight_control in enumerate(action):
            flight = self.sim.flights[i]
            x, y, alt, vel, hdng = flight.get_5_tuple()
            CTRL = self.IDX2ACTION[flight_control]
            if CTRL != 'NOOP':
                self.n_clearances += 1

            if CTRL == 'VEL_DEC' and vel > min_vel:
                flight.change_velocity(-self.sim.DVEL)
            elif CTRL == 'VEL_INC' and vel < max_vel:
                flight.change_velocity(self.sim.DVEL)
            elif CTRL == 'FL_DEC' and alt > min_alt:
                flight.change_altitude(-self.sim.DALT)
            elif CTRL == 'FL_INC' and alt < max_alt:
                flight.change_altitude(self.sim.DALT)
            elif CTRL != 'NOOP':
                self.n_invalids += 1

        self.cnt_clearances.append(self.n_clearances)
        self.invalids.append(self.n_invalids)

        '''
        Advancing Simulation by one time-step
        '''
        self.sim.advance_clock()

        self.current_state = self.extract_observation()

        reward = self.reward_function()

        self.cur_idx += 1

        term = False
        trunc = False
        # We're outatime doc!
        if self.cur_idx >= self.max_steps:
            reward += self.MAX_CLEARANCES
            trunc = True

        # You're just pressing all the buttons
        if self.n_clearances - self.n_invalids >= self.MAX_CLEARANCES:
            trunc = True

        # Retard. Retard. Retard.
        if self.n_invalids >= self.MAX_CLEARANCES * self.INVALID_MAXCLEARANCE_RATIO:
            reward = -self.MAX_CLEARANCES
            term = True

        return self.current_state, reward, term, trunc, {}

    def reward_function(self):
        _, coll_alts = self.sim.find_congestion(self.sim.flights)

        # Collisions coefficient is proportional to nb. of collisions
        coll_coeff = len(coll_alts)/(self.n - 1)
        coll_coeff *= self.COLLISION_COEFF_SCALE

        # Clearances coefficient
        clrn_coeff = self.n_clearances/self.MAX_CLEARANCES
        clrn_coeff *= self.CLEARANCE_COEFF_SCALE

        # Invalid actions coefficient
        invld_coeff = self.n_invalids/(self.MAX_CLEARANCES * self.INVALID_MAXCLEARANCE_RATIO)
        invld_coeff *= self.INVALID_COEFF_SCALE

        # Combine into one
        cur_complexity = -1. * (coll_coeff + clrn_coeff +  invld_coeff)

        return cur_complexity

    def render(self, save_path=None, fig=None, ax=None) -> RenderFrame | list[RenderFrame] | None:
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        render_sim = deepcopy(self.sim)
        for _ in range(self.cur_idx):
            render_sim.reverse_clock()

        # Add a colorbar next to the axis
        sm = mpl.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])  # This can be an empty array or any data array if needed
        ticks = render_sim.allowable_altitudes
        cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
        cbar.set_label('Altitude')

        # Optionally, you can set tick labels
        airspace_limit = render_sim.airspace_limit * 1.15
        ax.set_xlim(-airspace_limit, airspace_limit)
        ax.set_ylim(-airspace_limit, airspace_limit)
        hotspot_limit = render_sim.hotspot_limit
        ax.add_patch(Rectangle((-hotspot_limit, -hotspot_limit), width=2 * hotspot_limit, height=2 * hotspot_limit, alpha=0.7, fill=False, color='k'))

        arrow_size = 250
        fl_texts = []
        arrows = []
        flights = render_sim.flights
        for i, _ in enumerate(flights):
            c = render_sim.get_alt_color(flights[i].altitude)
            x, y, alt, vel, heading = flights[i].get_5_tuple()
            arrow = mpl.patches.FancyArrow(x, y, 0, 0, edgecolor=c, facecolor=c, head_width=arrow_size,
                                           head_length=arrow_size, zorder=15)

            arrows.append(arrow)
            text = ax.text(0, 0, '', fontsize=6)
            fl_texts.append(text)

        for lbl, i in self.ACTIONS2IDX.items():
            ax.scatter([], [], s=10, color=self.action_colors[i], marker=self.action_markers[i], label=lbl)
        ax.legend(loc='best', frameon=False, fontsize='small')

        def init():
            flights = render_sim.flights
            nonlocal fl_texts, arrows
            for i, flight in enumerate(flights):
                dx, dy = flight.get_heading_unit_vectors()
                arrow = mpl.patches.FancyArrow(flight.x, flight.y, dx, dy, width=0.01, color='b')
                ax.add_patch(arrow)
                arrows[i] = arrow
                fl_texts[i].set_text('')
            return fl_texts, arrows

        max_nb_cong = 0
        def update(frame):
            flights = render_sim.flights
            nonlocal fl_texts, arrows, max_nb_cong
            render_sim.advance_clock()
            for i, flight, in enumerate(flights):
                # Plot flights
                before_where = render_sim.flight_within_limit(flights[i])
                after_where = render_sim.flight_within_limit(flights[i])
                if before_where and not after_where:
                    arrows[i].remove()
                    fl_texts[i].remove()
                    continue
                elif not before_where or not after_where:
                    continue
                x, y, alt, vel, heading = flights[i].get_5_tuple()

                # Plot clearance
                clearance = self.clearances[frame][i]
                clearance_type = self.IDX2ACTION[clearance]
                if 'FL_' in clearance_type:
                    s = render_sim.norm_alt(alt) * 15 + 10
                elif 'VEL_' in clearance_type:
                    s = render_sim.norm_vel(vel) * 15 + 10
                else:
                    s = 10

                ax.scatter([x], [y], s=s, marker=self.action_markers[clearance],
                           color=self.action_colors[clearance])

                # Plot congestion volume
                cong_cen, cong_alt = render_sim.find_congestion(flights)
                max_nb_cong = max(max_nb_cong, len(cong_cen))
                for cen, calt in zip(cong_cen, cong_alt):
                    c_alt = render_sim.get_alt_color(calt)
                    ax.add_patch(plt.Circle(cen, hotspot_limit, color=c_alt, fill=False, alpha=0.1))

                arrow_size = 250
                ux, uy = flight.get_heading_unit_vectors()
                dx = ux * arrow_size
                dy = uy * arrow_size
                c = render_sim.get_alt_color(alt)
                arrows[i].remove()
                arrow = mpl.patches.FancyArrow(x, y, dx, dy, edgecolor=c, facecolor=c, head_width=arrow_size,
                                               head_length=arrow_size, zorder=15)
                ax.add_patch(arrow)
                arrows[i] = arrow

                fl_texts[i].remove()
                fl_texts[i] = ax.text(x, y, f'FL{alt:d}\nV{vel:d}', fontsize=7)
                s = f'Step = {frame}\nMax collisions = {max_nb_cong} Nb. Clearances = {self.cnt_clearances[frame]} Nb. Invalids = {self.invalids[frame]}'
                ax.set_title(s)
            return fl_texts, arrows

        # plt.legend()
        frames = np.arange(self.cur_idx)
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)

        if save_path is not None:
            ffwriter = mpl.animation.FFMpegWriter(fps=5)
            ani.save(save_path, dpi=100, writer=ffwriter)
        else:
            return ani

    def render_3d(self, save_path=None):
        """
            Plots flights as moving arrows in a 3D interactive animation.

            Parameters:
            flight_data (list of dicts): List containing flight data dictionaries with keys 'start', 'end', and 'color'.
                                         'start' and 'end' are tuples representing coordinates (x, y, z).
                                         'color' is a string representing the color of the arrow.
            """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        render_sim = deepcopy(self.sim)
        for _ in range(self.cur_idx):
            render_sim.reverse_clock()

        sm = mpl.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, ticks=render_sim.allowable_altitudes)
        cbar.set_label('Altitude')

        arrows = []
        lines = []
        history = []
        flights = render_sim.flights
        arrow_size = 2000
        quiver_kw = dict(arrow_length_ratio=0.1, pivot='tip')
        for i, flight in enumerate(flights):
            x, y, alt, vel, heading = flight.get_5_tuple()
            du, dv = flight.get_heading_unit_vectors()

            dx, dy = du * arrow_size, dv * arrow_size
            c = render_sim.get_alt_color(alt)
            arrow = ax.quiver(x, y, alt, dx, dy, 0., color=c, **quiver_kw)
            line, = ax.plot([],[],[], linestyle='--', c='k', lw=10)
            lines.append(line)
            history.append([x, y, alt])
            arrows.append(arrow)

        def init():
            ax.cla()
            # for _ in range(self.cur_idx):
            #     render_sim.reverse_clock()
            flights = render_sim.flights
            for i, flight in enumerate(flights):
                x, y, alt, vel, heading = flight.get_5_tuple()
                du, dv = flight.get_heading_unit_vectors()
                dx, dy = du * arrow_size, dv * arrow_size
                c = render_sim.get_alt_color(alt)

                arrow = ax.quiver(x, y, alt, dx, dy, 0, color=c, **quiver_kw)
                history[i] = [[x, y, alt]]
                line, = ax.plot([], [], [], linestyle='--', c='k', lw=1)
                lines[i] = line
                arrows[i] = arrow
            return arrows + lines

        def update(frame):
            # ax.cla()
            ax.set_title(frame)

            render_sim.advance_clock()
            flights = render_sim.flights
            for i, flight in enumerate(flights):
                x, y, alt, vel, heading = flight.get_5_tuple()
                du, dv = flight.get_heading_unit_vectors()

                dx, dy = du * arrow_size, dv * arrow_size
                c = render_sim.get_alt_color(alt)

                arrows[i].remove()
                arrow = ax.quiver(x, y, alt, dx, dy, 0, color=c, **quiver_kw)#,
                arrows[i] = arrow
                history[i].append([x, y, alt])
                h = np.array(history[i])
                lines[i].set_data(h[:,0],h[:,1])
                lines[i].set_3d_properties(h[:,2])

            centers, alts = render_sim.find_congestion(render_sim.flights)
            for c, a in zip(centers, alts):
                ax.scatter([c[0]],[c[1]],[a], marker='o', s=75*len(centers), c='r')
            ax.set_xlim(-10000, 10000)
            ax.set_ylim(-10000, 10000)
            ax.set_zlim(200, 500)
            return arrows + lines

        ani = animation.FuncAnimation(fig, update, frames=range(self.max_steps), init_func=init, interval=50, blit=False, repeat=True)

        if save_path is not None:
            ffwriter = mpl.animation.FFMpegWriter(fps=5)
            ani.save(save_path, dpi=100, writer=ffwriter)
        else:
            return ani


def test_collision_course_simulator():
    for n in np.arange(2, 8):
        for m in np.arange(2, 8):
            collision_sim = CollisionCourseSimulator(n=n, m=m)
            print(collision_sim)
            save_path = f'test_collision_n{n}m{m}_flights_{generate_random_alphanumeric(4)}.mp4'
            collision_sim.animate(save_path)
            print(f'Saved to {save_path}')

            collision_sim.save_flight_data('test.yml')
            coll_test = CollisionCourseSimulator.from_saved_data('test.yml')
            print(coll_test)
            exit(23)


def main():
    import pandas as pd
    env = UMFlightEnv(n=3, m=0)
    env.reset()
    print(repr(env))
    truncated = False
    zero_a = np.zeros(shape=env.action_space.shape)
    idx = 0
    dat = pd.DataFrame({'obs': [], 'action': [], 'reward': [], 'term': [], 'trunc': []})
    while not truncated:
        idx += 1
        a = zero_a
        otp1, r, terminated, truncated, _ = env.step(a)

        dat = dat.append({'obs': otp1, 'action': a, 'reward': r, 'term': terminated, 'trunc': truncated}, ignore_index=True)

    plt.plot(dat['reward'])
    plt.xlabel('Steps')
    plt.ylabel('Rewards')

    save_name = 'siunits_10km-airspace'
    # dat.to_csv(f'{save_name}.csv')

    # env.render(f'{save_name}.mp4')
    # env.render_3d_mayavi()
    # plt.show()


if __name__ == '__main__':
    main()
