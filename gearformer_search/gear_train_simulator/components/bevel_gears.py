import openmdao.api as om
import numpy as np

class BevelGears(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('place_coord_increm', types=int)
        self.options.declare('place_sign', types=int)
        self.options.declare('r_in', types=float)
        self.options.declare('r_out', types=float)
        self.options.declare('l_in', types=float)
        self.options.declare('l_out', types=float)
        self.options.declare('N_in', types=int)
        self.options.declare('N_out', types=int)

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

        r_in = self.options['r_in']
        r_out = self.options['r_out']
        l_in = self.options['l_in']
        l_out = self.options['l_out']
        N_in = self.options['N_in']
        N_out = self.options['N_out']
        place_coord_increm = self.options['place_coord_increm']
        place_sign = self.options['place_sign']

        pos_in = inputs['pos_in']
        flow_dir_in = inputs['flow_dir_in']
        motion_vector_in = inputs['motion_vector_in']
        q_dot_in = inputs['q_dot_in']

        place_dir = place_sign * np.roll(np.abs(flow_dir_in), place_coord_increm)

        outputs['pos_out'] = pos_in + (r_in + l_out/2) * place_dir + (l_in/2 + r_out) * flow_dir_in
        outputs['flow_dir_out'] = place_dir # will be ignored by next component
        outputs['motion_vector_out'] = -1 * place_dir * np.sum(motion_vector_in)
        outputs['q_dot_out'] = (q_dot_in * N_in) / N_out
