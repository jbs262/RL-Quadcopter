import numpy as np
import math
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_pose = init_pose
        self.success = False
        self.takeoff = False

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #original reward function: reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        thrusts = self.sim.get_propeler_thrust(self.sim.prop_wind_speed)
        linear_forces = self.sim.get_linear_forces(thrusts)
        distance = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        #speed = math.sqrt(np.square(self.sim.find_body_velocity()).sum())
        #with 300x300x300m env, the max distance from one corner to another is 519
        max_distance = 519
        #Focus quadcopter on not crashing but first rewarding an upward linear force until at the height of the target
        if self.sim.pose[2] < self.target_pos[2]:
            #velocity_discount = 1/speed
            reward = np.tanh(linear_forces[2])
        #after getting to the correct z-coordinate, move to the correct y-coordinate
        elif self.sim.pose[1] < self.target_pos[1]:
            #velocity_discount = 1/speed
            reward = 1 + np.tanh(linear_forces[1])
        #finally, after getting rewards for the x and y coordinates, give reward for distance
        #at this stage, the drone will have overshot the x and y coordinates, but it would be in a better area to
        #start searching for the x coordinate
        elif distance > 1 and self.sim.pose[2] > self.target_pos[2] and self.sim.pose[1] > self.target_pos[1] :
            reward = 2 + (1-math.pow((distance/300),.04))
        elif distance < 1:
            self.success = True
            reward = 100
        #possible reward for hover: np.exp(-np.square(linear_forces[2]))
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.takeoff = False
        self.success = False
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
