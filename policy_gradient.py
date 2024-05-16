import os
import torch
import numpy as np
from torch import nn
from torch import optim
import argparse
import collections
import functools
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence, List
import gym
import mujoco_py
from gym import utils
import torch.nn.functional as F
import copy
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
from pdb import set_trace as debug
from utils import rollout, log_density

def train_model(policy, baseline, trajs, policy_optim, baseline_optim, device, gamma=0.99, baseline_train_batch_size=64,
                baseline_num_epochs=5):

    states_all = []
    actions_all = []
    returns_all = []
    for traj in trajs:
        # Compute the return to go on the current batch of trajectories
        states_singletraj = traj['observations']
        actions_singletraj = traj['actions']
        rewards_singletraj = traj['rewards']
        returns_singletraj = np.zeros_like(rewards_singletraj)
        # disc_rewards = np.array([gamma ** i * rewards_singletraj[i] for i in range(len(rewards_singletraj))])
        # returns_singletraj = np.cumsum(disc_rewards[::-1])[::-1]
        for t in range(len(rewards_singletraj)):
            returns_singletraj[t] = sum([(gamma ** i) * rewards_singletraj[i] for i in range(t,len(rewards_singletraj))])
        states_all.append(states_singletraj)
        actions_all.append(actions_singletraj)
        returns_all.append(returns_singletraj)
    states = np.concatenate(states_all)
    actions = np.concatenate(actions_all)
    returns = np.concatenate(returns_all)

    # Normalize the returns by subtracting mean and dividing by std
    EPS = 1e-6
    returns = (returns - returns.mean()) / (returns.std() + EPS)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    for epoch in range(baseline_num_epochs):
        np.random.shuffle(arr)
        for i in range(n // baseline_train_batch_size):
            # Train baseline by regressing onto returns
            batch_index = arr[baseline_train_batch_size * i: baseline_train_batch_size * (i + 1)]
            return_batch = torch.from_numpy(returns[batch_index]).float().to(device)
            state_batch = torch.from_numpy(states[batch_index]).float().to(device)
            loss = criterion(baseline(state_batch), return_batch)
            baseline_optim.zero_grad()
            loss.backward()
            baseline_optim.step()


    # Train policy by optimizing surrogate objective: -log prob * (return - baseline)
    action, std, logstd = policy(torch.Tensor(states).to(device))
    log_policy = log_density(torch.Tensor(actions).to(device), policy.mu, std, logstd)
    baseline_pred = baseline(torch.from_numpy(states).float().to(device))
    returns_torch = torch.from_numpy(returns).float().to(device)
    # loss = -torch.dot(log_policy.reshape(-1), (returns_torch).reshape(-1))
    loss = -torch.dot(log_policy.reshape(-1), (returns_torch - baseline_pred).reshape(-1))
    policy_optim.zero_grad()
    loss.backward()
    policy_optim.step()

    del states, actions, returns, states_all, actions_all, returns_all


# Training loop for policy gradient
def simulate_policy_pg(env, policy, baseline, num_epochs=200, max_path_length=200, batch_size=100,
                       gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, device = "cuda", render=False):
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    rewards_all = []
    for iter_num in range(num_epochs):
        sample_trajs = []

        # Sampling trajectories
        for _ in range(batch_size):
            sample_traj = rollout(
                env,
                policy,
                episode_length=max_path_length,
                render=False)
            sample_trajs.append(sample_traj)

        # Logging returns
        rewards_np = rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
        rewards_all.append(rewards_np)
        if iter_num % print_freq == 0:
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)
        
    return rewards_all