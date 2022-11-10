import numpy as np
import pyglet
from pyglet import shapes
import time
import gym
from sklearn import preprocessing

# in pyglet the size unit is pixel
class PlaceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    viewer = None
    dt = 1    # refresh rate
    move_step = 8.4
    action_bound = [-8.4, 8.4]  # the movement range is between -3mm ~ 3mm -> 2.8 = 1mm
    # action_bound = [-5.6, 5.6]  # the movement range is between -3mm ~ 3mm -> 2.8 = 1mm
    goal_x = 300  # halb of the window's width
    goal_y = 300  # halb of the window's high
    gear_size = 112 # the radium of gear
    peg_size = 28 # the radium of peg, around 10cm

    press_force = 10    # press applied on the peg
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
        self.observation_space = gym.spaces.Box(
            np.array([-1.0,-1.0,0.0]),np.array([1.0,1.0,1.0])
        )
        
        self.old_torque = np.zeros(2,dtype=np.float32)  # record the last time torque sum
        

    def torque_cal(self,gear_info,on_goal):
        # calculate the torque in x and y axis which are set as reward
        # torque = force * distance
        # only when the distance between (0, 70) exit the torque (112/2 + 28/2)
        # when the gear outside the peg range no torque
        d_x = (self.goal_x - gear_info[0])/2.8
        d_y = (gear_info[1] - self.goal_y)/2.8

        distance = np.sqrt(d_x**2+d_y**2)
        if distance >= 0 and distance<=40:   # sqrt(25**2x2)=35.3
            self.press_force += np.random.normal(0,1)   # add random noise on press force

            torque_x = self.press_force * d_x  # around y axis
            torque_y = self.press_force * d_y  # around x axis/but in the real situation is different
        else:
            torque_x = 0
            torque_y = 0

        # state
        s = [torque_x,torque_y,on_goal]

        s = np.array(s,dtype=np.float32)

        # normalize the obs state -1~1, the max torque is press_force*40, 40 is max distance
        s = np.divide(s, (self.press_force * 40.))

        # torque_sum = abs(torque_x) + abs(torque_y)

        return s, d_x, d_y, distance
        
    def reset(self):
        self.i = 0
        self.total_reward = 0
        self.on_goal = 0    # whether stay on the target
        self.threshold = 0.3
        # random initial position of gear
        self.gear_info[0] = self.goal_x + np.random.uniform(low=-90.0,high=90.0)
        self.gear_info[1] = self.goal_y + np.random.uniform(low=-90.0,high=90.0)

        # self.gear_info[0] = self.goal_x + 112
        # self.gear_info[1] = self.goal_y 

        # calculate the torque as state
        s,_,_,_ = self.torque_cal(self.gear_info,self.on_goal)

        self.old_torque = abs(s[0]) + abs(s[1]) # save as the last time torque sum
        self.init_torque = self.old_torque  # record the initial torque sum for boudary detection
        return s

    def step(self, action):
        # time.sleep(1)
        self.i += 1 # calculate the step number
        done = False
        action = action*self.move_step # rescale to range -8.4 ~ 8.4 -> -3mm ~ 3mm
        action = np.clip(action, *self.action_bound)
        self.gear_info[0] += action[0] * self.dt    # x
        self.gear_info[1] += action[1] * self.dt    # y

        # calculate the torque as state
        s, d_x, d_y, distance = self.torque_cal(self.gear_info,self.on_goal)

        torque_sum = abs(s[0]) + abs(s[1])  # this sum is after normalize calculation

        step_r = 0
        if torque_sum >= self.old_torque:
            step_r = -1 # do not return
        else:
            step_r = (self.old_torque - torque_sum)

        o_t = self.old_torque
        # done and reward, gear center align with pey center
        # torque can not be used as condition since in the simulation no tolerance, absolute torque==0 is hard
        if  self.init_torque > o_t and distance<=self.threshold:
            step_r = (5. + (self.max_steps - self.i)*1.5)    # ealier reach goal that has more reward
            self.on_goal = 1
            done = True
        if self.i == self.max_steps:
            done = True
            step_r = -5
        if self.init_torque <= o_t and torque_sum==0:
            done = True
            step_r=-10.

        self.old_torque = torque_sum # update the old torque
        self.total_reward += step_r

        return s, step_r, done, {'i': self.i, 'x':self.gear_info[0],'y':self.gear_info[1],'d_x':d_x,'d_y':d_y,'t_s':torque_sum,'o_s':o_t,'i_s':self.init_torque,'distance':distance}

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