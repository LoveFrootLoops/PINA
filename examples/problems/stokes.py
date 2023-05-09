import numpy as np
import torch

from pina.problem import SpatialProblem
from pina.operators import nabla, grad, div
from pina import Condition, Span, LabelTensor


class Stokes(SpatialProblem):

    output_variables = ['ux', 'uy', 'p']
    spatial_domain = Span({'x': [-2, 2], 'y': [-1, 1]})

    def momentum(input_, output_):
        nabla_ = torch.hstack((LabelTensor(nabla(output_.extract(['ux']), input_), ['x']),
            LabelTensor(nabla(output_.extract(['uy']), input_), ['y'])))
        return - nabla_ + grad(output_.extract(['p']), input_)

    def continuity(input_, output_):
        return div(output_.extract(['ux', 'uy']), input_)

    def inlet(input_, output_):
        value = 2 * (1 - input_.extract(['y'])**2)
        return output_.extract(['ux']) - value

    def outlet(input_, output_):
        value = 0.0
        return output_.extract(['p']) - value

    def wall(input_, output_):
        value = 0.0
        return output_.extract(['ux', 'uy']) - value

    conditions = {
        'gamma_top': Condition(location=Span({'x': [-2, 2], 'y':  1}), function=wall),
        'gamma_bot': Condition(location=Span({'x': [-2, 2], 'y': -1}), function=wall),
        'gamma_out': Condition(location=Span({'x':  2, 'y': [-1, 1]}), function=outlet),
        'gamma_in':  Condition(location=Span({'x': -2, 'y': [-1, 1]}), function=inlet),
        'D': Condition(location=Span({'x': [-2, 2], 'y': [-1, 1]}), function=[momentum, continuity]),
    }
