
# -------


# -------

from dolfin    import *
from mshr      import *

# -------

# -------

# --------------------------------------------------------------

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


