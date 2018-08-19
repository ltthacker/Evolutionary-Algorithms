#!/usr/bin/env python3
import argparse
import gym
import time
import numpy as np

from lib import model, utils


MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001


def train(net, batch_noise, batch_reward, lr):
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    for p, p_update in zip(net.parameters, weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p += lr * update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    net = model.Net(env.observation_space.shape[0], env.action_space.n)

    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            pos_noise, neg_noise = utils.sample_noise(net)
            batch_noise.append(pos_noise)
            batch_noise.append(neg_noise)
            reward, steps = utils.eval_with_noise(env, net, pos_noise, args.noise_std)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = utils.eval_with_noise(env, net, neg_noise, args.noise_std)
            batch_reward.append(reward)
            batch_steps += steps
            if batch_steps > MAX_BATCH_STEPS:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > 199:
            print('Solved in {:d} steps'.format(step_idx))
            break

        train(net, batch_noise, batch_reward, args.lr)

        speed = batch_steps / (time.time() - t_start)
        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, m_reward, speed))

        
if __name__ == '__main__':
    main()