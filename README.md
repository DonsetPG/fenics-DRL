# fenics-DRL : 

Repository from the paper [A review on Deep Reinforcement Learning for Fluid Mechanics](https://arxiv.org/abs/1908.04127).

# How to use the code : 

## Install everything : 

### CFD : 

We used [Fenics](http://fenicsproject.org) for this project. The easiest way to install it is by using 
[Docker](https://docs.docker.com/install/). Then : 

```
docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current
```

should install Fenics. 

### DRL : 

We are using both [Gym - OpenAI](https://gym.openai.com) and [Stable-Baselines](https://github.com/hill-a/stable-baselines).
They both can installed with :

```
pip install gym 
pip install stable-baselines
``` 

More generally, everything you need can be installed with : 

```
pip install --user tensorflow keras gym stable-baselines sklearn
```

## Launch an experiment : 

An experiment consists of an Environement (based on [Gym - OpenAI](https://gym.openai.com) & [Fenics](http://fenicsproject.org)), and an Algorithm from [Stable-Baselines](https://github.com/hill-a/stable-baselines). 
They can be launched with [test_main.py](https://github.com/DonsetPG/fenics-DRL/test_main.py). You will only have to precise a few parameters : 
```python 
*nb_cpu*: the amount of CPU you want to use (e.g. 16)
*agents*: an array of the algorithms you want to use (e.g. ['PPO2','A2C'])
*name_env*: The name of the environment (e.g. 'Control-cylinder-v0')
*total_timesteps*: the amount of timesteps the training will last (e.g. 100000)
*text*: Some precisions you wanna add to you experiment (e.g. '1_step_1_episode_2CPU')
```

## Build your own environment : 

### The Gym.env environment : 

You can find examples of such environments in [example 1 : Control Cylinder](https://github.com/DonsetPG/fenics-DRL/examples/control_cylinder.py) or 
[example 2 : Flow Control Cylinder](https://github.com/DonsetPG/fenics-DRL/examples/control_flow_cylinder.py). They always share the same architecture : 

```python 
class FluidMechanicsEnv_(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                    **kwargs):
        ...
        self.problem = self._build_problem()
        self.reward_range = (-1,1)
        self.observation_space = spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float16)
        self.action_space = spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float16)
        
    def _build_problem(self,main_drag):
        ...
        return problem
        
        
    def _next_observation(self):
        ...
        
        
    def step(self, action):
        ...
        return obs, reward, done, {}
    
    def reset(self):
        ...
```

Here, most of these functions are DRL related, and more informations can be found at [this paper (for applications of DRL on fluid mechanics)](https://arxiv.org/abs/1908.04127)
or [here (for more general informations about DRL)](http://incompleteideas.net/book/the-book.html). The only link with Fenics is made with the 

```python
def _build_problem(self,main_drag):
        ...
        return problem
```

function, where you will be using functions from [Fenics](https://github.com/DonsetPG/fenics-DRL/deepfluid/fenics/). 

### Fenics functions : 

We built several functions to help you use Fenics and build DRL environment with it. Three main classes exist : 

- class Channel 
- class Obstacles
- class Problem

#### Channel : 

Allows you to create the 'box' where your simulation will take place. 

#### Obstacles : 

Allows you to add forms and obstacles (Circle, Square and Polygons) to your environment.

#### Problem : 

Build the simulation with Channl and Obstacles. Also get parameters for the mesh and the solver. Finally, this is  a Problem object you will return in the Gym.env class. 

# What's next : 

We built this repository in order to get a code as clean as possible for fluid mechanics with DRL. However, Fenics is not the best solver, especially with very demanding problem. The goal is to keep the same philosophy in mind (DRL and Fluid mechanics coupled easily) but with other (and faster)
libraries. 
Since most of these libraries are C++ based, and using powerful clusters, the architecture will be completely different. We are still working on it and doing our best to release an alpha version as soon as possible. 

This repository will be updated when such library finally comes out. Until then, we hope that with [this paper]() and this repository combined, some Fluid Mechanics researcher might want to try to apply Deep Reinforcement Learning to their experiments. 

# The Team : 

- [Paul Garnier](https://github.com/DonsetPG) : MINES Paristech - PSL Research University 
- [Jonathan Viquerat](https://github.com/jviquerat) : MINES Paristech - PSL Research University - CEMEF
- [Aurélien Larcher](https://github.com/alarcher) : MINES Paristech - PSL Research University - CEMEF
- [Elie Hachem](https://github.com/eliemines) : MINES Paristech - PSL Research University - CEMEF



