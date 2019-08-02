

# -------


import gym
import numpy as np
from gym import spaces

from fenics.fenics_domains import *
from utils import Encoder

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

We recreate here the experiment and the paper from --

We build a rectangle domain with a cylinder in it, and try to find the optimal position of another cylinder : a control cylinder

"""

n_envs = 0
encoding_dimension = 70
encoder = Encoder(encoding_dimension)

class FluidMechanicsEnv_(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                    **kwargs):

        global n_envs, encoding_dimension, encoder
        self.n_envs = n_envs
        n_envs += 1
        """

        We create here a gym.env in order to train DRL agents.

        **Input**:

        nb_episode: the number of episode we wanna train on
        T:          the horizon during an episode
        mesh_path:  folder we will create to save meshs during the training
        render_path:folder we will create to save imgs during training (if in render mode)
        X_LIM:      X-domain where we will create obstacles
        Y_LIM:      Y-domain where we will create obstacles
        problem:    the object used for simulations
        training:   if we are in training mode or not


        """

        super(FluidMechanicsEnv, self).__init__()

        self.nb_episode  =  kwargs.get('nb_episode',10)
        self.T =            kwargs.get('T',1)
        self.mesh_path =    kwargs.get('mesh_path','mesh')
        self.render_path =  kwargs.get('render_path','img')

        # Where our actions will take place :
        self.X_LIM = [-1.7,-0.7]
        self.Y_LIM = [-1.5,1.5]
        self.current_step = 0
        self.episode = 1

        # At the moment, our actions will only be the coordinates of a small cylinder :
        # We therefore need a X and a Y in [0,1] x [0,1]


        self.action_space = spaces.Box(
            low=np.array([-1.7,-1.5]), high=np.array([-0.7,1.5]), dtype=np.float16)

        #self.problem,self.main_drag = self._build_problem(main_drag=True)
        self.problem = self._build_problem(main_drag=False)

        self.training = True


        self.reward_range = (-15,15)
        if self.n_envs == 0:
            self.render = True
        # For the AutoEncoder :

        self.encoding_dimension = encoding_dimension

        self.encoder = encoder
        self.drag_avg = []


        # For the observations, we take the coordinate of the cylinder, and the last 3 positions of our small cylinder :
        # And we add also the encoding from the velocity fields
        #self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8+self.encoding_dimension,), dtype=np.float16)
        self.observation_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.float16)
        self.last_obstacle = None
        self.last_obstacle_ = None

    def _build_problem(self,main_drag):

        """

        We start by creating our main shape : a cylinder

        """

        #y_cylinder = np.random.uniform(-3,3,1)[0]

        forms = [
            ('Square',(0,0),0.5)
                ]

        nb_obstacles = 1



        action = self.action_space.sample()


        centers_X = [action[0]]
        centers_Y = [action[1]]

        obstacles = [

	        ]

        for i in range(nb_obstacles):

            radius = 0.1
            types = 'Circle'
            center = (centers_X[i],centers_Y[i])

            obstacles.append((types,center,radius))

        filename = 'log/img/'+'episode_'+str(self.episode)+'_timestep_'+str(self.current_step)+'_shape_'

        problem = Problem(render=self.render,
                    filename =filename,
                    min_bounds=(-3,-2),
                    max_bounds=(3,2),
                    types='Rectangle',
                    size_obstacles=nb_obstacles,
                    coords_obstacle = obstacles,
                    size_forms=1,
                    coords_forms = forms,
                    size_mesh = 100,
                    cfl=5,
                    final_time=10)
        problem.render = self.render
        problem.add_bottom_BC()
        problem.add_top_BC()
        problem.add_outflow_BC(style=0.0)
        problem.add_inflow_BC(style='Couette')

        #draft,lift,succes = problem.drag_lift()

        return problem


    def _next_observation(self):

        """

        Version 0.1.0 : We only send the shape and obstacle coordinates as observations for the agent and the velocity field

        """

        cylinder = self.problem.coords_forms[0][1]



        obs = np.array([cylinder[0]/10.,cylinder[1]/10.])


        return obs


    def _take_action(self, action):

        #cylinder = self.problem.coords_forms[0][1]

        """

        We build a new simulation with obstacles created regarding of the action chosen

        """


        center_X = action[0]
        center_Y = action[1]
        print('Cylindre en x := ',center_X)
        print('Cylindre en y := ',center_Y)
        forms = [
            ('Square',(0,0),0.5)
                ]

        obstacles = [
            ('Circle',(center_X,center_Y),0.1)
        ]

        filename = 'log/img/'+'episode_'+str(self.episode)+'_timestep_'+str(self.current_step)+'_shape_'

        problem = Problem(render=self.render,
                    filename = filename,
                    min_bounds=(-3,-2),
                    max_bounds=(3,2),
                    types='Rectangle',
                    size_obstacles=1,
                    coords_obstacle = obstacles,
                    size_forms=1,
                    coords_forms = forms,
                    size_mesh = 100,
                    cfl=5,
                    final_time=10)
        problem.render = self.render
        problem.add_bottom_BC()
        problem.add_top_BC()
        problem.add_outflow_BC(style=0.0)
        problem.add_inflow_BC(style='Couette')

        self.problem = problem


    def step(self, action):
        # Execute one time step within the environment

        self._take_action(action)
        self.current_step += 1

        if self.current_step > self.T:
            self.current_step = 0

        drag, lift, success = self.problem.drag_lift_navierstokes()

        reward = (drag) + 3.54 + 0.7
        if reward > 0:
            reward = reward*10

        print('Reward := ',reward)

        done = (self.current_step == self.T)

        obs = self._next_observation()

        return obs, reward, done, {}


    def reset(self):

        self.current_step = 0
        self.episode += 1
        self.problem = self._build_problem(main_drag = False)

        return self._next_observation()

    def render(self, mode='human', close=False):

        self.problem.render = True
        self.problem.render_mesh = True
        drag, lift, success = self.problem.drag_lift()
