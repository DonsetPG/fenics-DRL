
# -------

import math
import os

import matplotlib.pyplot    as plt
import numpy                as np
from dolfin import *
from mshr import *



# -------
# -------

# -------

# --------------------------------------------------------------

n_call = 0
u_0    = None
V_0    = None


class Channel:

    """
    Object computing where our problem will take place.
    At the moment, we restraint ourself to the easier domain possible : a rectangle.

    :min_bounds:    (tuple (x_min,y_min))   The lower corner of the domain
    :max_bounds:    (tuple (x_max,y_max))   The upper corner of the domain
    :Channel:       (Channel)               Where the simulation will take place
    """
    def __init__(self,
                min_bounds,
                max_bounds,
                type='Rectangle'):

            if len(min_bounds) == 2:
                self.x_min = min_bounds[0]
                self.y_min = min_bounds[1]
            else:
                raise ValueError('Min Bounds is supposed to be of length 2 but is of length ',len(min_bounds))

            if len(max_bounds) == 2:
                self.x_max = max_bounds[0]
                self.y_max = max_bounds[1]
            else:
                raise ValueError('Max Bounds is supposed to be of length 2 but is of length ',len(max_bounds))

            self.min = Point(self.x_min,self.y_min)
            self.max = Point(self.x_max,self.y_max)

            if type == 'Rectangle':
                self.channel = Rectangle(self.min,self.max)
            else:
                raise ValueError('Domains that are not Rectangle are not dealt with at the moment')

class Obstacles:

    """
    These are the obstacles we will add to our domain.
    You can add as much as you want.
    At the moment, we only take care of squares and circles and polygons

    :size_obstacles:    (int)           The amount of obstacle you wanna add
    :coords:            (python list)   List of tuples, according to 'fenics_utils.py : add_obstacle'
    :Obstacles:         (Obstacles)     An Obstacles object containing every shape we wanna add to our domain
    """


    def __init__(self,
                size_obstacles=0,
                coords = []):

        #Init :

        self.nb_obstacles = size_obstacles
        self.coord_obstacles = coords
        self.obstacles = []

        # Check values :

        if len(coords) == size_obstacles:
            for tup in coords:
                obs = create_obstacle(tup)
                self.obstacles.append(obs)

        else:
            raise ValueError('Warnings, nb_obstacles := ',size_obstacles,' but len(coords) := ', len(coords))


    def add_obstacle(self,
                    types,
                    center,
                    radius):
        """

        Method to add obstacle one by one :

        :types:     (String)        'Circle', 'Square' or 'Polygon'
        :center:    (int or array)  If type is Square or Circle, it's the center of the shape. Otherwise, it is the array of x coordinates of the polygon
        :radius:    (int or array)  If type is Square or Circle, it's the radius of the shape. Otherwise, it is the array of y coordinates of the polygon

        """
        obs = create_obstacle((types,center,radius))
        self.obstacles.append(obs)




class Problem:

    def __init__(self,
                min_bounds,
                max_bounds,
                import_mesh_path = None,
                types='Rectangle',
                size_obstacles=0,
                coords_obstacle = [],
                size_mesh = 64,
                cfl=5,
                final_time=20,
                meshname='mesh_',
                filename='img_',
                **kwargs):

        """
        We build here a problem Object, that is to say our simulation.

        :min_bounds:                (tuple      (x_min,y_min)) The lower corner of the domain
        :max_bounds:                (tuple      (x_max,y_max)) The upper corner of the domain
        :import:import_mesh_path:   (String)    If you want to import a problem from a mesh, and not built it with mshr
        :types:                     (String)    Forced to Rectangle at the moment
        :size_obstacle:             (int)       The amount of shape you wanna add to your domain
        :coords_obstacle:           (tuple)     the parameters for the shape creation
        :size_mesh:                 (int)       Size of the mesh you wanna create
        :cfl:                       (int)
        :final_time:                (int)       Length of your simulation
        :mesh_name:                 (String)    Name of the file you wanna save your mesh in
        :filename:                  (String)    Name of the file you wanna save the imgs in

        And in **kwargs :

        :reynolds:                  (float)     Reynolds parameter
        :v_in:                        --
        :mu:                          --
        :rho:                         --
        :sol_file:                  (String)    Name of the file you wanna save the .pvd file in
        :mesh and img _path:        (String)    Folder you want to use to save the mesh and the imgs
        :render and render_mesh:    (Bool)      If you want to save the mesh and the imgs
        :param:                     (Dict)      Allow to add more parameters if you want to
        """

        self.reynolds       =   kwargs.get('reynolds', 100.0)
        self.v_in           =   kwargs.get('v_in', 1.0 )
        self.mu             =   kwargs.get('mu', 1.0/self.reynolds)
        self.rho            =   kwargs.get('rho', 1.0)
        self.sol_file       =   kwargs.get('sol_file', 'shape.pvd')
        self.render_mesh    =   kwargs.get('render_mesh',False)
        self.mesh_path      =   kwargs.get('mesh_path','mesh')
        self.render         =   kwargs.get('render',False)
        self.img_path       =   kwargs.get('img_path','img')
        self.dt             =   kwargs.get('dt',0.01)
        self.param          =   kwargs.get('param',None)


        self.tag_shape = 5
        self.cfl = cfl
        self.final_time = final_time
        self.filename = filename
        self.meshname = meshname

        self.coords_obstacle = coords_obstacle



        """

        Channel: The domain where we compute our simulation

        """

        self.channel = Channel(min_bounds,
                                max_bounds,
                                types)



        """

        Obstacles we wanna add

        """

        self.obstacles = Obstacles(size_obstacles,
                                coords_obstacle)

        self.domain = self._build_domain()

        if import_mesh_path is not None:
            os.system("sed -i 's/dim=\"3\"/dim=\"2\"/g' "+import_mesh_path)
            self.mesh = Mesh(import_mesh_path)
        else:
            self.mesh = generate_mesh(self.domain, size_mesh)

        self.h = self.mesh.hmin()

        self.V = VectorFunctionSpace(self.mesh, 'CG', 2)
        self.Q = FunctionSpace(self.mesh, 'CG', 1)
        self.field = None
        """

        Bounds for the domain and the boundaries where we will compute the equations solution

        """

        self.xmax,self.xmin,self.ymax,self.ymin = self._build_bounds()

        self.bcu = []
        self.bcp = []

        if self.param is not None:

            for name_param in self.param:
                if name_param == 'jet_positions':
                    positions = self.param['jet_positions']
                if name_param == 'jet_radius':
                    radius = self.param['jet_radius']
                if name_param == 'jet_width':
                    width = self.param['jet_width']
            self.jets = []
            for pos in positions:
                self.jets.append(JetBCValue(radius, width, pos, Q=3., degree=1))

            shape = 'on_boundary && x[0]>(' + str(self.xmin) + ') && x[0]<' + str(self.xmax) + ' && x[1]>(' + str(self.ymin) + ') && x[1]<(' + str(self.ymax) + ')'
            for jet in self.jets:

                bc = DirichletBC(self.V, jet,shape)
                self.bcu.append(bc)

    def _build_domain(self):


        """

        Here, we start with the main channel, and then reduce
        it by taking out the shape we defined earlier (forms and obstacles )

        """

        domain = self.channel.channel
        for shapes in self.obstacles.obstacles:
            domain -= shapes


        return domain

    def _build_bounds(self):
        return self.channel.x_max,self.channel.x_min,self.channel.y_max,self.channel.y_min



        """

        Here, we will define several functions in order to define the boundary conditions of our problems,
        at the moment, we will allow :

        to set to 0 the top and bottom and outflow of our problem (or not)
        to set to a constant or couette flow the inflow

        """

    def add_bottom_BC(self,
                        val = 0.0):
        """

        define the bottom boundary conditions of our problem

        :val: (float) Forced value at the boundary condition

        """

        wall  = 'near(x[1], '+str(math.floor(self.ymin))+')'
        bcu_wall   = DirichletBC(self.V.sub(1), Constant(val), wall)
        self.bcu.append(bcu_wall)

    def add_top_BC(self,
                        val = 0.0):

        """

        define the top boundary conditions of our problem

        :val: (float) Forced value at the boundary condition

        """

        wall  = 'near(x[1], '+str(math.floor(self.ymax))+')'
        bcu_wall   = DirichletBC(self.V.sub(1), Constant(val), wall)
        self.bcu.append(bcu_wall)


    def add_outflow_BC(self,
                        style):

        """

        define the outflow boundary conditions of our problem

        :style: (float) Forced value at the boundary condition

        """

        outflow = 'near(x[0], '+str(math.floor(self.xmax))+')'

        if isinstance(style, str):
            raise ValueError('Type of flow not implemented')
        else:
            bcp_outflow = DirichletBC(self.Q,Constant(style),outflow)

        self.bcp.append(bcp_outflow)


    def add_inflow_BC(self,
                        style):

        """

        define the inflow boundary conditions of our problem

        :style: (String or float) Forced value at the boundary condition with Couette on constant flow

        """

        inflow  = 'near(x[0], '+str(math.floor(self.xmin))+')'
        if style == 'Couette':
            inflow_profile = constant_profile(self.mesh,degree=2)
            bcu_inflow  = DirichletBC(self.V,inflow_profile,inflow)
            self.bcu.append(bcu_inflow)
        elif isinstance(style, str):
            raise ValueError('Type of flow not implemented')
        else:
            bcu_inflow = DirichletBC(self.V,Constant((style, 0.0)), inflow)
            self.bcu.append(bcu_inflow)

    def update_problem(self,update,val_jet=0):
        """

        Allow you to update the problem during the resolution of any equations, with the parameters you want.

        :update:            (bool)  If we update it or not
        :args: ->
        """
        if update:
            for Q, jet in zip([val_jet,-val_jet], self.jets):
                jet.Q = jet.Q + 0.1*(Q - jet.Q)
            shape = 'on_boundary && x[0]>(' + str(self.xmin) + ') && x[0]<' + str(self.xmax) + ' && x[1]>(' + str(self.ymin) + ') && x[1]<(' + str(self.ymax) + ')'

            for i in range(len(self.jets)):
                jet = self.jets[i]
                self.bcu[i] = DirichletBC(self.V,jet,shape)



    def sigma(self,u, p):
        """

        Define the stress tensor

        :u: (Fenics field) Velocity field
        :p: (Fenics field) Pressure field

        """
        return 2.0*self.mu*epsilon(u) - p*Identity(len(u))


    def compute_drag_lift(self, u, p, mu, normal, gamma):
        eps = 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        sigma = 2.0 * mu * eps - p * Identity(len(u))
        traction = dot(sigma, normal)

        forceX = traction[0] * gamma
        forceY = traction[1] * gamma
        fX = assemble(forceX)
        fY = assemble(forceY)

        return (fX, fY)

    def drag_lift_navierstokes_init(self,update=False):

        """

        Compute drag and lift with respect of the Navier Stokes equation


        """


        x_lim_max,x_lim_min,y_lim_max,y_lim_min = self.xmax,self.xmin,self.ymax,self.ymin
        shape   = 'on_boundary && x[0]>('+str(x_lim_min)+') && x[0]<'+str(x_lim_max)+' && x[1]>('+str(y_lim_min)+') && x[1]<('+str(y_lim_max)+')'
        bcu_aile    = DirichletBC(self.V,Constant((0.0, 0.0)),  shape)
        self.bcu.append(bcu_aile)

        class Obstacle(SubDomain):
            def inside(self, x, on_boundary):

                return (on_boundary and
                    (x_lim_min < x[0] < x_lim_max) and
                    (y_lim_min < x[1] < y_lim_max))

        h    = self.mesh.hmin()

        # Compute timestep and max nb of steps
        dt             = self.dt
        self.timestep  = dt
        self.T         = self.final_time



        # Define output solution file
        obstacle = Obstacle()
        boundaries =  MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
        boundaries.set_all(0)
        obstacle.mark(boundaries, self.tag_shape)
        ds = Measure('ds', subdomain_data=boundaries)
        self.gamma_shape = ds(self.tag_shape)

        # Define trial and test functions
        u, v = TrialFunction(self.V), TestFunction(self.V)
        p, q = TrialFunction(self.Q), TestFunction(self.Q)

        # Define functions for solutions at previous and current time steps
        self.u_n, self.u_, self.u_m = Function(self.V), Function(self.V), Function(self.V)
        self.p_n, self.p_ = Function(self.Q), Function(self.Q)



        # Define expressions and constants used in variational forms
        U = 0.5*(self.u_n + u)
        self.n = FacetNormal(self.mesh)
        f = Constant((0, 0))
        self.dt = Constant(self.dt)
        self.mu = Constant(self.mu)
        self.rho = Constant(self.rho)

        # Set BDF2 coefficients for 1st timestep
        self.bdf2_a = Constant(1.0)
        self.bdf2_b = Constant(-1.0)
        self.bdf2_c = Constant(0.0)

        # Define variational problem for step 1
        # Using BDF2 scheme
        F1 = dot((self.bdf2_a*u + self.bdf2_b*self.u_n + self.bdf2_c*self.u_m)/self.dt,v)*dx + dot(dot(self.u_n, nabla_grad(u)), v)*dx + \
        inner(self.sigma(u, self.p_n), epsilon(v))*dx + dot(self.p_n*self.n, v) * \
        ds - dot(self.mu*nabla_grad(u)*self.n, v)*ds - dot(f, v)*dx
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Define variational problem for step 2
        self.a2 = dot(nabla_grad(p),   nabla_grad(q))*dx
        self.L2 = dot(nabla_grad(self.p_n), nabla_grad(q))*dx - (self.bdf2_a/self.dt)*div(self.u_)*q*dx

        # Define variational problem for step 3
        self.a3 = dot(u,  v)*dx
        self.L3 = dot(self.u_, v)*dx - (self.dt/self.bdf2_a)*dot(nabla_grad(self.p_ - self.p_n), v)*dx

        # Assemble A3 matrix since it will not need re-assembly
        self.A3 = assemble(self.a3)

        # Initialize drag and lift
        self.drag = 0.0
        self.lift = 0.0

        self.drag_inst = np.array([])
        self.lift_inst = np.array([])

        self.drag_avg = np.array([])
        self.lift_avg = np.array([])

        self.k = 0
        self.t = 0.0
        self.t_arr = np.array([])


    def drag_lift_navierstokes_step(self,num_steps=1,val_jet=0.0,update=False):

        ########################################
        # Time-stepping loop
        ########################################
        try:

            for m in range(num_steps):
                # Update current time
                self.update_problem(update,val_jet)
                self.t += self.timestep

                # Step 1: Tentative velocity step
                self.A1 = assemble(self.a1)
                self.b1 = assemble(self.L1)
                [bc.apply(self.A1) for bc in self.bcu]
                [bc.apply(self.b1) for bc in self.bcu]
                solve(self.A1, self.u_.vector(), self.b1, 'bicgstab', 'hypre_amg')

                # Step 2: Pressure correction step
                self.A2 = assemble(self.a2)
                self.b2 = assemble(self.L2)
                [bc.apply(self.A2) for bc in self.bcp]
                [bc.apply(self.b2) for bc in self.bcp]
                solve(self.A2, self.p_.vector(), self.b2, 'bicgstab', 'hypre_amg')

                # Step 3: Velocity correction step
                self.b3 = assemble(self.L3)
                solve(self.A3, self.u_.vector(), self.b3, 'cg', 'sor')

                # Update previous solution
                self.u_m.assign(self.u_n)
                self.u_n.assign(self.u_)
                self.p_n.assign(self.p_)

                # Compute and store drag and lift
                avg_start_it = math.floor(1)
                if (m > avg_start_it):
                    (fX, fY) = self.compute_drag_lift(self.u_, self.p_, self.mu, self.n, self.gamma_shape)
                    self.drag_inst = np.append(self.drag_inst, fX)
                    self.lift_inst = np.append(self.lift_inst, fY)
                    self.drag += fX
                    self.lift += fY
                    self.drag_avg = np.append(self.drag_avg, self.drag / (self.k + 1))
                    self.lift_avg = np.append(self.lift_avg, self.lift / (self.k + 1))
                    self.t_arr = np.append(self.t_arr, self.t)

                    # Increment local counter
                    self.k += 1

                    # Set BDF2 coefficients for m>1
                    self.bdf2_a.assign(Constant(3.0 / 2.0))
                    self.bdf2_b.assign(Constant(-2.0))
                    self.bdf2_c.assign(Constant(1.0 / 2.0))

            ########################################

            # Average drag and lift values
            uu = interpolate(self.u_n, self.V)
            v_ = uu.sub(1)
            self.field = v_.compute_vertex_values()
            drag = self.drag_avg[-1]
            lift = self.lift_avg[-1]

        except Exception as exc:
            print(exc)
            return 0.0, 0.0, False

        return drag, lift, True

    
    def drag_lift_navierstokes(self,update=False):

        self.drag_lift_navierstokes_init(update=update)
        num_steps = int(self.T/self.timestep)
        drag, lift, sucess = self.drag_lift_navierstokes_step(num_steps=num_steps)

        return drag,lift,sucess



"""

From Jean's code at the moment, needs to be rewrite bloody h
"""

def normalize_angle(angle):
    '''Make angle in [-pi, pi]'''
    assert angle >= 0

    if angle < pi:
        return angle
    if angle < 2*pi:
        return -((2*pi)-angle)

    return normalize_angle(angle - 2*pi)


class JetBCValue(UserExpression):


    def __init__(self, radius, width, theta0, Q, **kwargs):
        super().__init__(**kwargs)
        assert width > 0 and radius > 0
        theta0 = np.deg2rad(theta0)
        self.width = np.deg2rad(width)
        self.radius = radius
        self.theta0 = normalize_angle(theta0)

        self.Q = Q

    def eval(self, values, x):
        A = self.amplitude(x)
        xC = 0.
        yC = 0.

        values[0] = A*(x[0] - xC)
        values[1] = A*(x[1] - yC)

    def amplitude(self, x):
        theta = np.arctan2(x[1], x[0])

        # NOTE: motivation for below is cos(pi*(theta0 \pm width)/w) = 0 to
        # smoothly join the no slip.
        scale = self.Q/(2.*self.width*self.radius**2/pi)

        return scale*cos(pi*(theta - self.theta0)/self.width)

    # This is a vector field in 2d
    def value_shape(self):
        return (2, )



def create_obstacle(tup):
    """

    Allow to create fenics object that we will use in our simulation.
    Currently, we are dealing with the following shapes :
        - Circle
        - Square
        - Polygons

    :tup:           (tuple of length 3) cf below for more details **
    :obstacles:     (mshr object)       The shape we will add to our channel


    ** tup of length 3 :

    - For a circle :    ('Circle',center,radius)
    - For a square :    ('Square',center,radius)
    - For a polygon :   ('Polygon',[list of x_coordinates],[list of y_coordinates])

    """

    if len(tup) == 3:
        types = tup[0]
        if types == 'Circle':
            center = tup[1]
            radius = tup[2]
            return Circle(Point(center[0],center[1]),radius)
        elif types == 'Square':
            center = tup[1]
            radius = tup[2]
            bottom = Point(center[0] - radius, center[1] - radius)
            top = Point(center[0] + radius, center[1] + radius)
            return Rectangle(bottom,top)
        elif types == 'Polygon':
            vect_points = []
            for x_,y_ in zip(tup[1],tup[2]):
                vect_points.append(Point(x_,y_))
            return Polygon(vect_points)
        else:
            raise ValueError('Obstacles that are not Squares, Circles or Polygon are not dealt with at the moment')
    else:
        raise ValueError('Tuple is supposed to be of length 3 but is of length ',len(tup))



def constant_profile(mesh, degree):
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]

    H = top - bot

    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                       '0'), bot=bot, top=top, H=H, Um=Um, degree=degree, time=0)



    # Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))




