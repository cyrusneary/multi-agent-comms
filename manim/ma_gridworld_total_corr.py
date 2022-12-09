from tkinter import Image
from manim import *
from matplotlib.pyplot import pink
from scipy.fftpack import shift
import numpy as np
import pickle
import manimpango

class CreateCircle(Scene):
    def construct(self):

        # Load the trajectory data
        # save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05_full_comm.pkl'
        # save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05_imag_play.pkl'
        # save_file_name = '2022-03-30-14-28-33_ma_gridworld_reachability_add_end_state_0p05_imag_play.pkl'
        save_file_name = '2022-03-30-14-28-33_ma_gridworld_reachability_add_end_state_0p05_full_comm.pkl'

        save_file_path = os.path.join(os.path.curdir, 'trajectory_data')
        save_str = os.path.join(save_file_path, save_file_name)
        with open(save_str, 'rb') as f:
            data = pickle.load(f)

        ground_color = '#A5A871'
        water_color = '#586494'
        mountain_color = '#8C8188'
        background_color = '#EBEBEB'

        grid_color = LIGHT_GRAY

        env_data = data['env_data']
        trajectory_list = data['trajectory_list']

        # Set some scene parameters
        width = env_data['Nc']
        height = env_data['Nr']
        grid_xstep = 1.0
        grid_ystep = 1.0

        time_between_steps = 0.05
        animation_time = 0.5

        # setup the scene
        self.camera.background_color = background_color

        # Setup the containing gridworld
        grid = Rectangle(width=width, 
                        height=height, 
                        color=grid_color, 
                        grid_xstep=grid_xstep, 
                        grid_ystep=grid_ystep)
        grid.set_fill(ground_color, opacity=0.5)

        # setup the mountains
        mountains = env_data['walls']
        mountain_1 = Rectangle(width=grid_xstep, 
                                height=grid_ystep,
                                color=mountain_color)
        mountain_1_row, mountain_1_col = mountains[0]
        mountain_1.set_fill(mountain_color, opacity=1.0)

        mountain_2 = Rectangle(width=grid_xstep, 
                                height=grid_ystep,
                                color=mountain_color)
        mountain_2_row, mountain_2_col = mountains[1]
        mountain_2.set_fill(mountain_color, opacity=1.0)

        mountain_3 = Rectangle(width=grid_xstep, 
                                height=grid_ystep,
                                color=mountain_color)
        mountain_3_row, mountain_3_col = mountains[2]
        mountain_3.set_fill(mountain_color, opacity=1.0)

        # setup the water
        water = env_data['lava']
        water_1 = Rectangle(width=grid_xstep, 
                                height=grid_ystep,
                                color=water_color)
        water_1_row, water_1_col = water[0]
        water_1.set_fill(water_color, opacity=1.0)

        water_2 = Rectangle(width=grid_xstep, 
                                height=grid_ystep,
                                color=water_color)
        water_2_row, water_2_col = water[1]
        water_2.set_fill(water_color, opacity=1.0)

        water_3 = Rectangle(width=grid_xstep, 
                                height=grid_ystep,
                                color=water_color)
        water_3_row, water_3_col = water[2]
        water_3.set_fill(water_color, opacity=1.0)


        # Instantiate the robot mobjects
        a1 = ImageMobject('assets/robot_pink.png')
        a1.scale(0.2)
        a0 = ImageMobject('assets/robot_blue.png')
        a0.scale(0.2)

        fire = ImageMobject('assets/fire.png')
        fire.scale(0.1)

        # Instantiate the targets
        target_0 = Circle(radius=grid_xstep/2, color=BLUE_D)
        target_1 = Circle(radius=grid_xstep/2, color=PINK)

        target_0.set_fill(BLUE_D, opacity=0.5)
        target_1.set_fill(PINK, opacity=0.5)

        # Define some helpful functions for the scene
        def row_col_to_manim_vec(row : int, col : int) -> np.ndarray:
            """
            Transform (row, col) matrix-based gridworld coordinates to the
            (x,y,z) vector coordinates used by manim for plotting.
            """
            y = -(row - height/2 + grid_ystep/2)
            x = col - width/2 + grid_xstep/2
            z = 0.0
            return np.array([x,y,z])

        def reinit_scene(init_state, traj_ind):
            # Setup the agents
            a0_row, a0_col, a1_row, a1_col = init_state
            a0_pos = row_col_to_manim_vec(a0_row, a0_col)
            a1_pos = row_col_to_manim_vec(a1_row, a1_col)
            a0.move_to(a0_pos)
            a1.move_to(a1_pos)

            # Setup the targets
            target_0_row, target_0_col, target_1_row, target_1_col = env_data['target_states'][0]
            target_0_pos = row_col_to_manim_vec(target_0_row, target_0_col)
            target_1_pos = row_col_to_manim_vec(target_1_row, target_1_col)

            target_0.move_to(target_0_pos)
            target_1.move_to(target_1_pos)
        
            # Setup the mountains
            mountain_1_pos = row_col_to_manim_vec(mountain_1_row, mountain_1_col)
            mountain_2_pos = row_col_to_manim_vec(mountain_2_row, mountain_2_col)
            mountain_3_pos = row_col_to_manim_vec(mountain_3_row, mountain_3_col)
            mountain_1.move_to(mountain_1_pos)
            mountain_2.move_to(mountain_2_pos)
            mountain_3.move_to(mountain_3_pos)

            # setup the water
            water_1_pos = row_col_to_manim_vec(water_1_row, water_1_col)
            water_2_pos = row_col_to_manim_vec(water_2_row, water_2_col)
            water_3_pos = row_col_to_manim_vec(water_3_row, water_3_col)
            water_1.move_to(water_1_pos)
            water_2.move_to(water_2_pos)
            water_3.move_to(water_3_pos)

            # Setup the trajectory text
            text = Text(
                "Path #{}".format(traj_ind), 
                font='Sans',
                color=BLACK, 
                font_size=30,
                weight=manimpango.Weight.SEMIBOLD.name
            )
            text.move_to(np.array([0.0, height/2 + grid_ystep/2, 0.0]))

            # Animate the creation of the scene
            self.play(Create(grid), 
                        Create(target_0), 
                        Create(target_1),
                        Create(mountain_1),
                        Create(mountain_2),
                        Create(mountain_3),
                        Create(water_1),
                        Create(water_2),
                        Create(water_3),
                        Write(text))
            self.add(a0)
            self.add(a1)

            return text

        def deconstruct_scene(text):
            # self.remove(a0, a1)
            self.play(FadeOut(a0), 
                        FadeOut(a1), 
                        FadeOut(grid),
                        FadeOut(target_0),
                        FadeOut(target_1),
                        FadeOut(mountain_1),
                        FadeOut(mountain_2),
                        FadeOut(mountain_3),
                        FadeOut(water_1),
                        FadeOut(water_2),
                        FadeOut(water_3),
                        FadeOut(text))

        # Iterate over trajectories and animate them.
        ordered_traj_ind_list = [8, 1, 9]
        display_count = 1
        for traj_ind in ordered_traj_ind_list:# range(len(trajectory_list)):
            traj = trajectory_list[traj_ind]
            text = reinit_scene(traj[0], display_count)
            display_count = display_count + 1
            self.wait(time_between_steps)
            for t in range(1,len(traj)):
                a0_next_row, a0_next_col, a1_next_row, a1_next_col = traj[t]
                a0_next_pos = row_col_to_manim_vec(a0_next_row, a0_next_col)
                a1_next_pos = row_col_to_manim_vec(a1_next_row, a1_next_col)
                self.play(a0.animate.move_to(a0_next_pos), a1.animate.move_to(a1_next_pos))

                # If they crash add a nice animation
                if (a0_next_pos == a1_next_pos).all():
                    fire.move_to(a0_next_pos)
                    self.add(fire)
                    self.play(fire.animate.scale(3.5))
                    self.wait(1.0)
                    self.remove(fire)
                    fire.scale(1/3.5)

            deconstruct_scene(text)
            self.wait(time_between_steps)