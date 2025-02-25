import openmdao.api as om
import json
from .components.spur_gears import SpurGears
from .components.shaft import Shaft
from .components.bevel_gears import BevelGears
from .components.worm_wheel import WormWheel
from .components.wheel_worm import WheelWorm
from .components.pinion_hypoid import PinionHypoid
from .components.hypoid_pinion import HypoidPinion
from .components.pinion_rack import PinionRack
from .components.rack_pinion import RackPinion

class GearSystem(om.Group):

    def initialize(self):
        self.options.declare('input_data', types=dict)

    def setup(self):
        input_data = self.options['input_data']
        seq = input_data["gear_train_sequence"]

        with open("/Users/cheongh/git/gear_train_EA/gear_train_simulator/components/catalogue.json", 'r') as file:
            components = json.load(file)

        comp_idx = 0
        subsys_idx = 0
        subsys_track = []
        while comp_idx < len(seq):
            if "translate" in seq[comp_idx]:
                if seq[comp_idx].split("_")[1] == "plus":
                    place_sign = 1
                else:
                    place_sign = -1
                shaft_type = seq[comp_idx+1]

                r = components[shaft_type]["outside_dia"] / 2
                if shaft_type != "SH-*":
                    l = components[shaft_type]["total_length"]
                else:
                    if subsys_idx == 0:
                        l1 = 0.0
                    else:
                        prev_comp = seq[comp_idx-1]
                        if "MHP1" in prev_comp and "L" in prev_comp:
                            l1 = 0.0
                        else:
                            l1 = components[prev_comp]["total_length"]/2
                    if seq[comp_idx+2] == "<end>":
                        l2 = 0.0
                    else:
                        next_comp = seq[comp_idx+2]
                        if "MHP1" in next_comp and "L" in next_comp:
                            l2 = 0.0
                        else:
                            l2 = components[next_comp]["total_length"]/2
                    l = l1 + l2

                self.add_subsystem(name='comp'+str(subsys_idx), subsys=Shaft(place_sign=place_sign, r=r, l=l),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx+1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx+1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx+1))],
                                   )
                subsys_track.append("shaft")
                comp_idx += 2
                subsys_idx += 1

            elif "MRGF" in seq[comp_idx] and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1

                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]

                N_rack = components[gear1_type]["no_of_teeth"]
                l_rack = components[gear1_type]["total_length"]
                h_rack = components[gear1_type]["height"]
                r_pinion = components[gear2_type]["pitch_dia"] / 2
                N_pinion = components[gear2_type]["no_of_teeth"]

                self.add_subsystem(name='comp' + str(subsys_idx),
                                   subsys=RackPinion(place_coord_increm=place_coord_increm, place_sign=place_sign,
                                                     r_pinion=r_pinion, N_pinion=N_pinion, N_rack=N_rack, l_rack=l_rack,
                                                     h_rack=h_rack),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                   )
                subsys_track.append("rack_pinion")
                comp_idx += 2
                subsys_idx += 1

            elif "MSGA" in seq[comp_idx] and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1

                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]

                if "MSGA" in gear2_type:
                    r_in = components[gear1_type]["pitch_dia"] / 2
                    r_out = components[gear2_type]["pitch_dia"] / 2
                    N_in = components[gear1_type]["no_of_teeth"]
                    N_out = components[gear2_type]["no_of_teeth"]

                    self.add_subsystem(name='comp'+str(subsys_idx),
                                       subsys=SpurGears(place_coord_increm=place_coord_increm, place_sign=place_sign,
                                                        r_in=r_in, r_out=r_out, N_in=N_in, N_out=N_out),
                                       promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                        ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                        ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                        ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                       promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                         ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                         ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                         ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                       )
                    subsys_track.append("spur_gears")
                    comp_idx += 2
                    subsys_idx += 1

                elif "MRGF" in gear2_type:
                    r_pinion = components[gear1_type]["pitch_dia"] / 2
                    N_pinion = components[gear1_type]["no_of_teeth"]
                    N_rack = components[gear2_type]["no_of_teeth"]
                    l_rack = components[gear2_type]["total_length"]
                    h_rack = components[gear2_type]["height"]

                    self.add_subsystem(name='comp' + str(subsys_idx),
                                       subsys=PinionRack(place_coord_increm=place_coord_increm, place_sign=place_sign,
                                                         r_pinion=r_pinion, N_pinion=N_pinion, N_rack=N_rack,
                                                         l_rack=l_rack, h_rack=h_rack),
                                       promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                        ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                        ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                        ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                       promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                         ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                         ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                         ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                       )
                    subsys_track.append("pinion_rack")
                    break

            elif ("MMSG" in seq[comp_idx] or "SBSG" in seq[comp_idx]) and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1
                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]
                r_in = components[gear1_type]["pitch_dia"] / 2
                r_out = components[gear2_type]["pitch_dia"] / 2
                l_in = components[gear1_type]["total_length"]
                l_out = components[gear2_type]["total_length"]
                N_in = components[gear1_type]["no_of_teeth"]
                N_out = components[gear2_type]["no_of_teeth"]

                self.add_subsystem(name='comp' + str(subsys_idx),
                                   subsys=BevelGears(place_coord_increm=place_coord_increm, place_sign=place_sign,
                                                     r_in=r_in, r_out=r_out, N_in=N_in, N_out=N_out,
                                                     l_in=l_in, l_out=l_out),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                   )
                subsys_track.append("bevel_gears")
                comp_idx += 3
                subsys_idx += 1

            elif "SWG" in seq[comp_idx] and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1
                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]
                l_worm = components[gear1_type]["total_length"]
                r_worm = components[gear1_type]["pitch_dia"] / 2
                N_worm = components[gear1_type]["no_of_teeth"]
                r_wheel = components[gear2_type]["pitch_dia"] / 2
                N_wheel = components[gear2_type]["no_of_teeth"]

                self.add_subsystem(name='comp' + str(subsys_idx),
                                   subsys=WormWheel(place_coord_increm=place_coord_increm, place_sign=place_sign,
                                                     l_worm=l_worm, r_worm=r_worm, r_wheel=r_wheel, N_worm=N_worm,
                                                    N_wheel=N_wheel),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                   )
                subsys_track.append("worm_wheel")
                comp_idx += 3
                subsys_idx += 1

            elif "AG" in seq[comp_idx] and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1
                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]
                r_wheel = components[gear1_type]["pitch_dia"] / 2
                N_wheel = components[gear1_type]["no_of_teeth"]
                l_worm = components[gear2_type]["total_length"]
                r_worm = components[gear2_type]["pitch_dia"] / 2
                N_worm = components[gear2_type]["no_of_teeth"]

                self.add_subsystem(name='comp' + str(subsys_idx),
                                   subsys=WheelWorm(place_coord_increm=place_coord_increm, place_sign=place_sign,
                                                     l_worm=l_worm, r_worm=r_worm, r_wheel=r_wheel, N_worm=N_worm,
                                                    N_wheel=N_wheel),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                   )
                subsys_track.append("wheel_worm")
                comp_idx += 3
                subsys_idx += 1

            elif "MHP" in seq[comp_idx] and "R" in seq[comp_idx] and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1
                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]
                N_pinion = components[gear1_type]["no_of_teeth"]
                md_pinion = components[gear1_type]["mounting_distance"]
                l_pinion = components[gear1_type]["total_length"]
                N_hypoid = components[gear2_type]["no_of_teeth"]
                r_hypoid = components[gear2_type]["pitch_dia"] / 2
                offset = components[gear2_type]["offset"]

                self.add_subsystem(name='comp' + str(subsys_idx),
                                   subsys=PinionHypoid(place_coord_increm=place_coord_increm,
                                                       place_sign=place_sign,
                                                       N_pinion=N_pinion, l_pinion=l_pinion, md_pinion=md_pinion,
                                                       N_hypoid=N_hypoid, md_hypoid=r_hypoid, offset=offset),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                   )
                subsys_track.append("pinion_hypoid")
                comp_idx += 3
                subsys_idx += 1

            elif "MHP" in seq[comp_idx] and "L" in seq[comp_idx] and "mesh" in seq[comp_idx+1]:
                mesh_info = seq[comp_idx+1].split("_")

                if "1st" in mesh_info[1]:
                    place_coord_increm = 1
                elif "2nd" in mesh_info[1]:
                    place_coord_increm = 2
                if "plus" in mesh_info[2]:
                    place_sign = 1
                elif "minus" in mesh_info[2]:
                    place_sign = -1
                gear1_type = seq[comp_idx]
                gear2_type = seq[comp_idx+2]
                N_pinion = components[gear2_type]["no_of_teeth"]
                l_pinion = components[gear2_type]["total_length"]
                md_pinion = components[gear2_type]["mounting_distance"]
                N_hypoid = components[gear1_type]["no_of_teeth"]
                md_hypoid = components[gear1_type]["mounting_distance"]
                offset = components[gear1_type]["offset"]

                self.add_subsystem(name='comp' + str(subsys_idx),
                                   subsys=HypoidPinion(place_coord_increm=place_coord_increm,
                                                       place_sign=place_sign,
                                                       N_pinion=N_pinion, l_pinion=l_pinion, md_pinion=md_pinion,
                                                       N_hypoid=N_hypoid, md_hypoid=md_hypoid, offset=offset),
                                   promotes_inputs=[('pos_in', 'pos_' + str(subsys_idx)),
                                                    ('flow_dir_in', 'flow_dir_' + str(subsys_idx)),
                                                    ('motion_vector_in', 'motion_vector_' + str(subsys_idx)),
                                                    ('q_dot_in', 'q_dot_' + str(subsys_idx))],
                                   promotes_outputs=[('pos_out', 'pos_' + str(subsys_idx + 1)),
                                                     ('flow_dir_out', 'flow_dir_' + str(subsys_idx + 1)),
                                                     ('motion_vector_out', 'motion_vector_' + str(subsys_idx + 1)),
                                                     ('q_dot_out', 'q_dot_' + str(subsys_idx + 1))],
                                   )
                subsys_track.append("hypoid_pinion")
                comp_idx += 3
                subsys_idx += 1

            else:
                comp_idx += 1

        # print(subsys_track)

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()