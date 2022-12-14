'''
The env for using continous policy like ddpg, td3
'''
import numpy as np
import pyglet
from pyglet import shapes
import time
import gym
import cv2
import os

# in pyglet the size unit is pixel
class PlaceEnvImage(gym.Env):
    #metadata = {'render.modes': ['ansi']}
    #metadata = {'render.modes': ['human']}


    viewer = None
    #viewer = 1

    move_step = 8.4
    action_bound = [-8.4, 8.4]  # the movement range is between -3mm ~ 3mm -> 2.8 = 1mm
    # action_bound = [-5.6, 5.6]  # the movement range is between -3mm ~ 3mm -> 2.8 = 1mm
    goal_x = 300  # halb of the window's width
    goal_y = 300  # halb of the window's high
    gear_size = 112 # the radium of gear
    peg_size = 28 # the radium of peg, around 10cm

    # press_force = 10    # press applied on the peg
    max_steps = 20 # max step pre epoch

    def __init__(self):
        # save the coordinate of gear center point
        self.gear_info = np.zeros(2,dtype=np.float32)
        self.gear_info[0] = 200 # x
        self.gear_info[1] = 200 # y
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,),dtype = np.float32
        )
        # self.observation_space = gym.spaces.Box(
        #     low=-1., high=1., shape=(2,),dtype = np.float32
        # )
        # self.observation_space = gym.spaces.Box(
        #     0, 255, (84, 84), dtype='uint8' # [height, width, 3]
        # )
        self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        # spaces = {
        #     # Here's an observation space for 84 wide x 84 high greyscale image inputs:
        #     'image': gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
        #     # And the x and y coordinate of gear will be set as state too
        #     'position': gym.spaces.Box(low=-600, high=600, shape=(2,), dtype=np.float32)
        # }

        # self.observation_space = gym.spaces.Dict(spaces)

        # self.old_torque = np.zeros(2,dtype=np.float32)  # record the last time torque sum
        self.image_path = os.path.join(os.getcwd(),"images")

    def state_cal(self,gear_info):
        # calculate the difference between goal center and gear center -> find the image frame
        # 1mm = 5.6
        d_x = (gear_info[0] - self.goal_x)
        d_y = (self.goal_y - gear_info[1])

        distance = np.sqrt(d_x**2+d_y**2)

        # index for reading image
        # the center point is (4,12), the point distance in row is 2mm(11.2), in colum is 5mm(28)
        x = round(4 - d_y/28)
        y = round(12 - d_x/11.2)

        if x<=7 and y<=24 and x>=0 and y>=0:   
            # read a image as state from corresponding position (chose the closest one)
            img_name = str(x) + "_" + str(y) + '.png'
            img_path = os.path.join(self.image_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
            img = cv2.resize(img, (84,84), interpolation= cv2.INTER_LINEAR) # resize the image
            img = np.expand_dims(img, -1)   # reshape the shape -> grayscale only has one channel (84,84) -> need to extend one dim
        else:
            img = np.zeros((84,84,1), dtype='uint8')

        # create observation: the capture image and the position of the gear
        # s = {
        #     'image': img,
        #     'position': np.array(gear_info)
        # }

        return img, d_x, d_y, distance
        
    def reset(self):
        self.i = 0
        self.total_reward = 0
        # self.on_goal = 0    # whether stay on the target
        self.threshold = 0.3
        # random initial position of gear
        self.gear_info[0] = self.goal_x + np.random.uniform(low=-90.0,high=90.0)
        self.gear_info[1] = self.goal_y + np.random.uniform(low=-90.0,high=90.0)

        # self.gear_info[0] = self.goal_x + 112
        # self.gear_info[1] = self.goal_y 

        # image frame as state
        s,_,_,_ = self.state_cal(self.gear_info)

        return s

    def step(self, action):
        # time.sleep(1)
        self.i += 1 # calculate the step number
        done = False
        action = action*self.move_step # rescale to range -8.4 ~ 8.4 -> -3mm ~ 3mm
        action = np.clip(action, *self.action_bound)
        self.gear_info[0] += action[0]  # x
        self.gear_info[1] += action[1]  # y

        # calculate the torque as state
        s, d_x, d_y, distance = self.state_cal(self.gear_info)

        step_r = 1/(self.i + 1)

        distance /= 5.6
        # done and reward, gear center align with pey center
        # torque can not be used as condition since in the simulation no tolerance, absolute torque==0 is hard
        if  distance<=self.threshold:
            step_r = (5. + (self.max_steps - self.i)*1.5)    # ealier reach goal that has more reward
            done = True
        if self.i == self.max_steps:
            done = True
            step_r = -5
        if distance>120:
            done = True
            step_r=-10.

        self.total_reward += step_r

        return s, step_r, done, {'i': self.i, 'x':self.gear_info[0],'y':self.gear_info[1],'d_x':d_x,'d_y':d_y,'distance':distance}

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.gear_info)
            #pass
        self.viewer.render()

    def sample_action(self):
        action = np.zeros(2,dtype=np.float32)
        action[0] = np.random.uniform(low=-self.move_step,high=self.move_step)
        action[1] = np.random.uniform(low=-self.move_step,high=self.move_step)
        return action
    
    def close(self):
        return super().close()

class Viewer(pyglet.window.Window):
    win_w = 600 # window size width
    win_h = 600 # window size high

    peg_size = 28 # around 10cm
    gear_size = 112 # around 40cm

    def __init__(self, gear_info):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=self.win_w, height=self.win_h, resizable=False, caption='Placement', vsync=False)
        # background color
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.gear_info = gear_info

        self.batch = pyglet.graphics.Batch()    # display whole batch at once, put all components in one batch. if render, only render batch is enough
        self.peg = shapes.Circle(self.win_w/2,self.win_h/2,self.peg_size,color=(50, 225, 30),batch=self.batch)
        self.gear = shapes.Circle(200,200,self.gear_size,color=(250, 25, 30),batch=self.batch)
        self.gear.opacity = 150
        self.gear_circle = shapes.Circle(200,200,self.peg_size,color=(255, 182, 193),batch=self.batch)
        self.gear_circle.opacity = 150

    def render(self):
        self._update_gear()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
    
    def on_draw(self):
        self.clear()
        self.batch.draw()
    
    def _update_gear(self):
        x = self.gear_info[0]
        y = self.gear_info[1]

        self.gear.x = x
        self.gear.y = y

        self.gear_circle.x = x
        self.gear_circle.y = y

if __name__ == '__main__':
    env = PlaceEnv()
    # while True:
    #     s = env.reset()
    #     env.render()
    #     time.sleep(0.5)
    #     for i in range(400):
    #         env.render()
    #         s,_,_,info=env.step(env.sample_action())
    #         print("state:{},d_x:{},d_y:{},step_r:{},i:{}".format
    #                 (s,info['d_x'],info['d_y'],info['sr'],info['i']))
    env.reset()
    env.render()
    x = 10
    y = -10

    x = 0
    # y = 0
    while x<=112:
        env.render()
        time.sleep(0.5)
        action = np.array([x,y],dtype=np.float32)
        s,_,_,info=env.step(action)
        print("state:{},d_x:{},d_y:{},step_r:{},i:{}".format
        (s,info['d_x'],info['d_y'],info['sr'],info['i']))
