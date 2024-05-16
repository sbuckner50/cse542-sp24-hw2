import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Policy Gradient
with open('data/pg_rewards.npy', 'rb') as f:
    pg_rewards_base = np.load(f)
with open('data/pg_rewards_alt.npy', 'rb') as f:
    pg_rewards_nobase = np.load(f)

sns.set_theme()
fig = plt.figure()
plt.plot(range(len(pg_rewards_base)), pg_rewards_base, color='r', label='baselined', linewidth=2)
plt.plot(range(len(pg_rewards_nobase)), pg_rewards_nobase, color='b', label='no baseline', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.legend()
plt.savefig("figures/pg_analysis.pdf")

## Actor-Critic
with open('data/ac_rewards.npy', 'rb') as f:
    ac_rewards_target = np.load(f)
with open('data/ac_rewards_alt.npy', 'rb') as f:
    ac_rewards_notarget = np.load(f)

sns.set_theme()
fig = plt.figure()
plt.plot(range(len(ac_rewards_target)), ac_rewards_target, color='r', label='target Q-network', linewidth=2)
plt.plot(range(len(ac_rewards_notarget)), ac_rewards_notarget, color='b', label='current Q-network', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.legend()
plt.savefig("figures/ac_analysis.pdf")