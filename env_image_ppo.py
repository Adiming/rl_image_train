import numpy as np
import pyglet
from pyglet import shapes
import time
import gym
import cv2
import os
from random import randrange
from collections import deque

ACTION_MAP = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

# in pyglet the size unit is pixel
class PlaceEnvImage(gym.Env):
    metadata = {'render.modes': ['human']}

    #viewer = None

    move_x = 5.6
    move_y = 5.6

    goal_x = 300  # halb of the window's width
    goal_y = 300  # halb of the window's high
    gear_size = 112 # the radium of gear
    peg_size = 28 # the radium of peg, around 10cm

    max_steps = 50 # max step pre epoch

    def __init__(self):
        # save the coordinate of gear center point
        self.gear_info = np.zeros(2,dtype=np.float32)
        self.gear_info[0] = 200 # x
        self.gear_info[1] = 200 # y
        self.action_map = ACTION_MAP
        
        self.action_space = gym.spaces.Box(
        low=-1, high=1, shape=(2, ), dtype=np.float32)
       
        # self.observation_space = gym.spaces.Box(
        # low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

        self.stack_size = 4 # We stack 4 frames

        self.image_path = os.path.join(os.getcwd(),"images")
        
    def stack_frames(self,stacked_frames, state, is_new_episode):
        # Preprocess frame
        # frame = preprocess_frame(state)
        frame = state
        
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((84,84), dtype=np.uint8) for i in range(self.stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)
            
        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2) 
        
        return stacked_state, stacked_frames

    def state_cal(self,gx,gy):
        y = gx # the sequency is opposite
        x = gy

        if x<30 and y<35 and x>=0 and y>=0:   
            # read a image as state from corresponding position (chose the closest one)
            img_name = str(x) + "_" + str(y) + '.png'
            img_path = os.path.join(self.image_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
            img = cv2.resize(img, (84,84), interpolation= cv2.INTER_LINEAR) # resize the image
            # img = np.expand_dims(img, -1)   # reshape the shape -> grayscale only has one channel (84,84) -> need to extend one dim
        else:
            # img = np.zeros((84,84,1), dtype=np.uint8)
            img = np.zeros((84,84), dtype=np.uint8)

        return img
        
    def reset(self):
        self.i = 0
        self.total_reward = 0
        # random initial position of gear
        gx = randrange(30)
        gy = randrange(35)

        self.x = gx
        self.y = gy

        self.gear_info[0] = self.goal_x + (gx - 14)*self.move_x
        self.gear_info[1] = self.goal_y + (gy - 17)*self.move_y

        # self.gear_info[0] = self.goal_x + 12*11.2
        # self.gear_info[1] = self.goal_y 
        
        # Initialize deque with zero-images one array for each image
        self.s  =  deque([np.zeros((84,84), dtype=np.uint8) for i in range(self.stack_size)], maxlen=4)

        # image frame as state
        # s = self.state_cal(gx,gy)
        img = self.state_cal(gx,gy)
        new_episode = True
        _,self.s = self.stack_frames(stacked_frames=self.s,state=img,is_new_episode=new_episode)

        return self.s

    def step(self, action):
        # every step is between -0.005~0.005, one image is 0.001 step
        action = action*5 
        
        gx = self.x + round(action[0])
        gy = self.y + round(action[1])
        # time.sleep(1)
        self.i += 1 # calculate the step number
        done = False

        # img as state
        # s = self.state_cal(gx,gy)
        img = self.state_cal(gx,gy)
        new_episode = False
        _,self.s = self.stack_frames(stacked_frames=self.s,state=img,is_new_episode=new_episode)

        # step_r = 0
        step_r = 1 / (abs(gx-14)+abs(gy-17)+1)

        if  gx==14 and gy==17:
            step_r = (1. + (self.max_steps - self.i))    # ealier reach goal that has more reward
            done = True
        if self.i == self.max_steps:
            done = True
            step_r = -5
        if gy>35 or gx>30 or gy<0 or gx<0:
            done = True
            step_r=-10.

        self.total_reward += step_r
        return self.s, step_r, done, {'i': self.i, 'x':self.gear_info[0],'y':self.gear_info[1],'g_x':gx,'g_y':gy}

    def render(self):
        if self.viewer is None:
            #self.viewer = Viewer(self.gear_info)
            #self.viewer.render()
            pass

    def sample_action(self):
        action = randrange(5)
        return action
    
    def close(self):
        return super().close()
'''
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
'''
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
