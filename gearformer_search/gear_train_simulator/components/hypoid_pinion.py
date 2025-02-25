import openmdao.api as om
import numpy as np

class HypoidPinion(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('place_coord_increm', types=int)
        self.options.declare('place_sign', types=int)
        self.options.declare('N_pinion', types=int)
        self.options.declare('l_pinion', types=float)
        self.options.declare('md_pinion', types=float)
        self.options.declare('N_hypoid', types=int)
        self.options.declare('md_hypoid', types=float)
        self.options.declare('offset', types=float)

    def setup(self):
        self.add_input('pos_in', shape=3, desc='input position', units='m')
        self.add_input('flow_dir_in', shape=3, desc='input motion flow direction')
        self.add_input('motion_vector_in', shape=3, desc='rotation/translation vector')
        self.add_input('q_dot_in', shape=1, desc='input speed', units='rad/s')

        self.add_output('pos_out', shape=3, desc='output position', units='m')
        self.add_output('flow_dir_out', shape=3, desc='output motion flow  direction')
        self.add_output('motion_vector_out', shape=3, desc='rotation/translation vector')
        self.add_output('q_dot_out', shape=1, desc='output speed', units='rad/s')

    def setup_partials(self):
        self.declare_partials('pos_out', 'pos_in', method='fd')
        self.declare_partials('pos_out', 'flow_dir_in', method='fd')
        self.declare_partials('flow_dir_out', 'flow_dir_in', method='fd')
        self.declare_partials('motion_vector_out', 'flow_dir_in', method='fd')
        self.declare_partials('motion_vector_out', 'motion_vector_in', method='fd')
        self.declare_partials('q_dot_out', 'q_dot_in', method='fd')

    def compute(self, inputs, outputs):

        N_pinion = self.options['N_pinion']
        N_hypoid = self.options['N_hypoid']
        l_pinion = self.options['l_pinion']
        md_pinion = self.options['md_pinion']
        md_hypoid = self.options['md_hypoid']
        offset = self.options['offset']
        place_coord_increm = self.options['place_coord_increm']
        place_sign = self.options['place_sign']

        pos_in = inputs['pos_in']
        flow_dir_in = inputs['flow_dir_in']
        motion_vector_in = inputs['motion_vector_in']
        q_dot_in = inputs['q_dot_in']

        place_dir = place_sign * np.roll(np.abs(flow_dir_in), place_coord_increm)
        flow_dir = np.cross(place_dir, flow_dir_in)

        outputs['pos_out'] = pos_in + md_hypoid * flow_dir_in + (md_pinion - l_pinion/2) * flow_dir + offset * place_dir
        outputs['flow_dir_out'] = flow_dir
        outputs['motion_vector_out'] = -1 * np.cross(place_dir, motion_vector_in)
        outputs['q_dot_out'] = (q_dot_in * N_hypoid) / N_pinion

