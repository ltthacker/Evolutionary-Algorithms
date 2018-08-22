#!/usr/bin/env python3
import argparse
import collections
import copy
import gym
import roboschool
import sys
import time
import numpy as np

import torch
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import model, utils


ENVIRONMENT = 'RoboschoolHalfCheetah-v1'

NOISE_STD = 0.005
PARENTS_COUNT = 10
WORKERS_COUNT = 4
SEEDS_PER_WORKER = 500
MAX_SEED = 2 ** 32 - 1


def evaluate(env, net, device='cpu'):
    obs = env.reset()
    reward = 0.0
    steps = 0
    done = False
    while not done:
        obs = torch.FloatTensor([obs]).to(device)
        actions = net(obs)
        action = actions.data.cpu().numpy()[0]
        obs, r, done, _ = env.step(action)
        reward += r
        steps += 1
    return reward, steps


def mutate_net(net, seed, noise_std, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += noise_std * noise
    return new_net


def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = model.CheetahNet(env.observation_space.shape[0], env.action_space.shape[0])
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps'])


def worker_func(input_queue, output_queue):
    env = gym.make(ENVIRONMENT)
    cache = {}
    
    while True:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for seeds in parents:
            if len(seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, seeds[-1])
                else:
                    net = build_net(env, seeds)
            else:
                net = build_net(env, seeds)
        new_cache[seeds] = net
        reward, steps = evaluate(env, net)
        output_queue.put(OutputItem(seeds=seeds, reward=reward, steps=steps))
    cache = new_cache


def get_elite(elites):
    elite = None
    top_reward = 0
    env = gym.make(ENVIRONMENT)
    for e in elites:
        reward = 0
        seeds = e[0]
        net = build_net(env, seeds)
        for _ in range(15):
            r, _ = evaluate(env, net)
            reward += r
        mean_reward = reward / 15.
        if mean_reward > top_reward:
            elite = seeds
            top_reward = mean_reward
    return elite


def main():
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--noise_std', type=float, default=NOISE_STD)
    args = parser.parse_args()
    writer = SummaryWriter(comment='-cheetah-ga-batch')
    device = 'cuda' if args.cuda else 'cpu'

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    print('All started')

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            seeds, reward, steps = output_queue.get()
            population.append((seeds, reward))
            batch_steps += steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]

        utils.write_ga(gen_idx, rewards, gen_seconds, batch_steps, speed, writer)
        print('%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s' % (gen_idx, reward_mean, reward_max, reward_std, speed))
        
        elite = get_elite(population[:5])


if __name__ == '__main__':
    main()
