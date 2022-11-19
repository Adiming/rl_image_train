import numpy as np
import pyglet
from pyglet import shapes
import time
import gym
import cv2
import os
from random import randrange

ACTION_MAP = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

# in pyglet the size unit is pixel
class PlaceEnvImage(gym.Env):
    metadata = {'render.modes': ['human']}

    viewer = None
    move_x = 11.2
    move_y = 28

    goal_x = 300  # halb of the window's width
    goal_y = 300  # halb of the window's high
    gear_size = 112 # the radium of gear
    peg_size = 28 # the radium of peg, around 10cm

    max_steps = 20 # max step pre epoch

    def __init__(self):
        # save the coordinate of gear center point
        self.gear_info = np.zeros(2,dtype=np.float32)
        self.gear_info[0] = 200 # x
        self.gear_info[1] = 200 # y
        self.action_map = ACTION_MAP
        # up, down, left, right
        self.action_space = gym.spaces.Discrete(4)
       
        self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.image_path = os.path.join(os.getcwd(),"images")

    def state_cal(self,gx,gy):
        y = gx # the sequency is opposite
        x = gy

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

        return img
        
    def reset(self):
        self.i = 0
        self.total_reward = 0
        self.threshold = 0.3
        # random initial position of gear
        gx = randrange(25)
        gy = randrange(8)

        self.x = gx
        self.y = gy

        self.gear_info[0] = self.goal_x + (gx - 12)*11.2
        self.gear_info[1] = self.goal_y + (gy - 4)*28

        # self.gear_info[0] = self.goal_x + 12*11.2
        # self.gear_info[1] = self.goal_y 

        # image frame as state
        s = self.state_cal(gx,gy)

        return s

    def step(self, action):
        if isinstance(action, str) and action in ('up', 'down', 'left', 'right'):
            pass
        if isinstance(action, (int, np.int64, np.int32,np.ndarray)):
            action = self.action_map[int(action)]
        else:
            print(action, type(action))
            raise
        
        if action == 'up':
            self.y += 1
            self.gear_info[1] += self.move_y
        elif action == 'down':
            self.y -= 1
            self.gear_info[1] -= self.move_y
        elif action == 'left':
            self.x += 1
            self.gear_info[0] -= self.move_x
        elif action == 'right':
            self.x -= 1
            self.gear_info[0] += self.move_x
        
        gx = self.x
        gy = self.y
        # time.sleep(1)
        self.i += 1 # calculate the step number
        done = False

        # calculate the torque as state
        s = self.state_cal(gx,gy)

        step_r = 0

        if  gx==12 and gy==4:
            step_r = (1. + (self.max_steps - self.i))    # ealier reach goal that has more reward
            done = True
        if self.i == self.max_steps:
            done = True
            step_r = -5
        if gy>7 and gx>24 and gy<0 and gx<0:
            done = True
            step_r=-10.

        self.total_reward += step_r
        return s, step_r, done, {'i': self.i, 'x':self.gear_info[0],'y':self.gear_info[1],'g_x':gx,'g_y':gy}

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.gear_info)
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
    env = PlaceEnvImage()
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