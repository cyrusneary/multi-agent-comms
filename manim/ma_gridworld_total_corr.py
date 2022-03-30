from manim import *
from matplotlib.pyplot import pink
from scipy.fftpack import shift
import numpy as np
import pickle



class CreateCircle(Scene):
    def construct(self):

        # Load the trajectory data
        save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05.pkl'
        save_file_path = os.path.join(os.path.curdir, 'trajectory_data')
        save_str = os.path.join(save_file_path, save_file_name)
        with open(save_str, 'rb') as f:
            data = pickle.load(f)

        env_data = data['env_data']
        trajectory_list = data['trajectory_list']

        # Set some scene parameters
        width = env_data['Nc']
        height = env_data['Nr']
        grid_xstep = 1.0
        grid_ystep = 1.0

        time_between_steps = 0.05

        # setup the scene
        self.camera.background_color = WHITE

        grid = Rectangle(width=width, height=height, color=BLACK, grid_xstep=grid_xstep, grid_ystep=grid_ystep)
        self.add(grid)

        # Instantiate the robot mobjects
        a1 = ImageMobject('assets/robot_pink.png')
        a1.scale(0.2)
        a0 = ImageMobject('assets/robot_blue.png')
        a0.scale(0.2)

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

        def reinit_scene(init_state):
            a0_row, a0_col, a1_row, a1_col = init_state
            a0_pos = row_col_to_manim_vec(a0_row, a0_col)
            a1_pos = row_col_to_manim_vec(a1_row, a1_col)
            a0.move_to(a0_pos)
            a1.move_to(a1_pos)

            target_0_row, target_0_col, target_1_row, target_1_col = env_data['target_states'][0]
            target_0_pos = row_col_to_manim_vec(target_0_row, target_0_col)
            target_1_pos = row_col_to_manim_vec(target_1_row, target_1_col)

            target_0.move_to(target_0_pos)
            target_1.move_to(target_1_pos)
        
            self.play(Create(grid), Create(target_0), Create(target_1))
            self.add(a0)
            self.add(a1)

        def deconstruct_scene():
            # self.remove(a0, a1)
            self.play(FadeOut(a0), 
                        FadeOut(a1), 
                        FadeOut(grid),
                        FadeOut(target_0),
                        FadeOut(target_1))

        # for traj in trajectory_list:
        traj = trajectory_list[2]
        reinit_scene(traj[0])
        self.wait(time_between_steps)
        for t in range(1,len(traj)):
            a0_next_row, a0_next_col, a1_next_row, a1_next_col = traj[t]
            a0_next_pos = row_col_to_manim_vec(a0_next_row, a0_next_col)
            a1_next_pos = row_col_to_manim_vec(a1_next_row, a1_next_col)
            self.play(a0.animate.move_to(a0_next_pos), a1.animate.move_to(a1_next_pos))

        deconstruct_scene()
        self.wait(time_between_steps)

        # reinit_scene(trajectory_list[0][0])
        # self.wait(time_between_steps)
        # deconstruct_scene()
        # self.wait(1)

        # a0_pos = row_col_to_manim_vec(0, 0)

        # a0.move_to(a0_pos)
        # self.add(a0)

        # print(a0.get_center())

        # a0_pos = a0.get_center()        # pos_transform_x = pos[1] - width/2
        # a0_pos = (a0_pos - [height/2, width/2, 0.0])

        # # pos_transform_y = -(pos[0] - width/2)

        # self.add(a1)
        # self.wait(0.1)

        # self.play(a1.animate.shift(UP), a0.animate.move_to(np.array([1.0,0.0,0.0])))

        # self.wait(0.1)
        # self.play(a1.animate.shift(LEFT))
        # self.wait(0.1)