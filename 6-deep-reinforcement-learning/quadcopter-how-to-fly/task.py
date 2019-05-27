import numpy as np
from physics_sim import PhysicsSim
from scipy.stats import norm

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
        self.max_duration = 10.0
        # Goal is to stay in this position, it will start at 0.0.0 and it should stay stable at
        #0.0.10. It is no allowed to move in x and y directions and it is not allowed to rotate.
        # it should go from 0.0.0 to 0.0.10 with little variations on phi, theta and psi.
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        #reward gaussian distribution
        sigma_values = [0.45]
        mu = 0
        x = np.linspace(-10, 10, 100)
        self.dist = norm(mu, sigma_values)
        sigma_values = [3]
        mu = 10
        self.dist_z = norm(mu, sigma_values)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #rewards regarding velocity
        #reward = 2 * self.sim.pose[2]
        reward = self.dist.pdf(self.sim.v[0])[0] * 25 #reward for staying still
        reward += self.dist.pdf(self.sim.v[1])[0] * 25 #reward for staying still
        #if ((self.sim.pose[2] < self.target_pos[2] and self.sim.v[2] > 0.0) or 
        #   (self.sim.pose[2] > self.target_pos[2] and self.sim.v[2] < 0.0)):
        #reward = self.dist_z.pdf(self.sim.pose[2])
        #reward -= 0.005*((self.sim.v[0])**2+(self.sim.v[1])**2)
        #reward -= 0.1*(abs(self.sim.angular_v).sum())
        #reward += 5
        reward += 1.25 * self.sim.pose[2]
        if self.sim.v[2] > 0:
            #reward += 3 * self.sim.pose[2]
            reward += 25
        #if abs(self.sim.pose[2]-self.target_pos[2])<(self.sim.pose[2]-self.target_pos[2])/2:
        #    reward += 10
        #    reward += 20
        #reward = np.tanh(1 - 0.003*abs(self.sim.pose[:3] - self.target_pos)).sum()
        if (self.sim.pose[2] == self.target_pos[2] and self.sim.v[2] == 0.0):
            reward += 1000 #reward for going up and centered on 10
        #if ((self.sim.pose[2] < self.target_pos[2] and self.sim.v[2] < 0.0) or 
        #   (self.sim.pose[2] > self.target_pos[2] and self.sim.v[2] > 0.0)):
            #reward += - self.dist_z.pdf(self.sim.pose[2]) * 300.0
        #    reward -= 20
        #rewards regarding angles
        #reward += self.dist.pdf(self.sim.pose[3] * 10)[0] * 20.0 #reward for staying still
        #reward += self.dist.pdf(self.sim.pose[4] * 10)[0] * 20.0 #reward for staying still
        #reward += self.dist.pdf(self.sim.pose[5] * 10)[0] * 20.0 #reward for staying still

        #reward regarding position
        #reward += self.dist.pdf(self.sim.pose[0])[0] * 40.0 #reward for staying still
        #reward += self.dist.pdf(self.sim.pose[1])[0] * 40 #reward for staying still

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            #if done:
            #    reward += 10
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state