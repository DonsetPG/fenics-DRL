# -------


import gym
import numpy as np
from gym import spaces

from deepfluid.fenics.fenics_domains import Problem
from deepfluid.utils import Encoder

# -------
# -------

# -------

# --------------------------------------------------------------


# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------- Work in Progress -----------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------


"""

We recreate here the experiment and the paper from https://arxiv.org/abs/1808.07664

We build a rectangle domain with a cylinder in it, and try to find a control strategies for active flow control

"""

n_envs = 0
encoding_dimension = 70
encoder = Encoder(encoding_dimension)


class FluidMechanicsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 **kwargs):

        global n_envs, encoding_dimension, encoder
        self.n_envs = n_envs
        n_envs += 1

        super(FluidMechanicsEnv, self).__init__()

        self.T = kwargs.get('T', 80)
        self.mesh_path = kwargs.get('mesh_path', 'mesh')
        self.render_path = kwargs.get('render_path', 'img')

        # Where our actions will take place :
        self.X_LIM = [-2, 20]
        self.Y_LIM = [-2, 2]
        self.current_step = 0
        self.episode = 1

        # At the moment, our actions will only be the coordinates of a small cylinder :
        # We therefore need a X and a Y in [0,1] x [0,1]

        self.action_space = spaces.Box(
            low=np.array([-5]), high=np.array([5]), dtype=np.float16)

        # self.problem,self.main_drag = self._build_problem(main_drag=True)
        self.problem = self._build_problem(main_drag=False)

        self.training = True

        self.reward_range = (-15, 15)
        if self.n_envs == 0:
            self.render = True
        # For the AutoEncoder :

        self.encoding_dimension = encoding_dimension
        self.encoder = encoder
        self.drag_avg = []

        # For the observations, we take the last action made :
        # And we add also the encoding from the velocity fields
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + self.encoding_dimension,), dtype=np.float16)
        # self.observation_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.float16)
        self.last_obstacle = None
        self.last_obstacle_ = None

    def _build_problem(self, main_drag):

        """

        We start by creating our main shape : a cylinder

        """

        # y_cylinder = np.random.uniform(-3,3,1)[0]

        forms = [
            ('Circle', (0, 0), 0.5)
        ]

        nb_obstacles = 1

        filename = 'log_JR/img/' + 'episode_' + str(self.episode) + '_timestep_' + str(self.current_step) + '_shape_'

        param = {'jet_positions': [90, 270], 'jet_radius': 0.05, 'jet_width': 10}

        problem = Problem(render=self.render,
                          filename=filename,
                          min_bounds=(-2, -2),
                          max_bounds=(20, 2),
                          types='Rectangle',
                          size_obstacles=1,
                          coords_obstacle=forms,
                          size_mesh=90,
                          cfl=0.8,
                          final_time=20,
                          param=param)

        problem.render = self.render
        problem.add_bottom_BC()
        problem.add_top_BC()
        problem.add_outflow_BC(style=0.0)
        problem.add_inflow_BC(style='Couette')
        problem.drag_lift_navierstokes_init(update=True)
        drag, lift,sucess = problem.drag_lift_navierstokes_step(num_steps=50,val_jet=3.0,update=True)
        print('Init drag := ',drag)
        indx = 0

        #for _ in range(10):
            #drag, lift, indx = problem.drag_lift_ns_step(0, indx, update=True)


        return problem

    def _next_observation(self):

        """

        Version 0.1.0 : We only send the velocity field and the time and the last action

        """

        time = self.problem.t / 20.
        jet = self.problem.jets[0].Q


        if self.problem.field is None:
            field_encoded = np.zeros((self.encoding_dimension,))
        else:
            field_encoded = self.encoder._get_encoding(self.problem.field)

        obs = np.array([time, jet])

        obs = np.append(obs, field_encoded)

        return obs

    def step(self, action):
        # Execute one time step within the environment

        val_jet = action[0]
        print('val jet := ',val_jet)
        drag, lift,sucess = self.problem.drag_lift_navierstokes_step(num_steps=50, val_jet=3.0, update=True)
        field = self.problem.field
        if field is not None:
            self.encoder._train(field)

        self.current_step += 1

        if self.current_step > self.T:
            self.current_step = 0

        reward = drag + 1.159 - 0.2 * abs(lift)

        print("reward:= ",reward, '- Drag:= ',drag, '- Lift:= ',lift)

        done = (self.current_step == self.T)

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):

        self.current_step = 0
        self.episode += 1
        self.problem = self._build_problem(main_drag=False)

        return self._next_observation()

    def render(self, mode='human', close=False):

        self.problem.render = True
        self.problem.render_mesh = True
        drag, lift, success = self.problem.drag_lift()
