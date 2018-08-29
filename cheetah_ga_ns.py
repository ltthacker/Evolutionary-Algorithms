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
# Population size = 2000 (WORKERS_COUNT * SEEDS_PER_WORKER)
WORKERS_COUNT = 4
SEEDS_PER_WORKER = 500
MAX_SEED = 2 ** 32 - 1

NOVEL_START = 0.05
NOVEL_GEN = 400


def euclidean_distance(x, y):
    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)


def compute_novelty_vs_archive(archive, novelty_vector, k=25):
    distances = []
    nov = novelty_vector.astype(np.float)
    for point in archive:
        distances.append(euclidean_distance(point.astype(np.float), nov))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indices = (distances).argsort()[:k]
    top_k = distances[top_k_indices]
    return top_k.mean()


def get_mean_bc(env, net, tslimit, num_rollouts=1):
    novelty_vector = []
    for n in range(num_rollouts):
        _, _, nv = evaluate(env, net)
        novelty_vector.append(nv)
    return np.mean(novelty_vector, axis=0)


def get_pos(self, model):
    mass = model.body_mass
    xpos = model.data.xipos
    center = (np.sum(mass * xpos, 0) / np.sum(mass))
    return center[0], center[1]    


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

    x_pos, y_pos = get_pos(env.unwrapped.model)
    novelty_vector = np.array([x_pos, y_pos])
    return rewards, steps, novelty_vector


def mutate_net(net, seed, noise_std, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += noise_std * noise
    return new_net


def build_net(env, seeds)
    torch.manual_seed(seeds[0])
    net = model.CheetahNet(env.observation_space.shape[0], env.action_space.shape[0])
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps', 'novelty_vector'])


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
        reward, steps, nv = evaluate(env, net)
        output_queue.put(OutputItem(seeds=seeds, reward=reward, steps=steps, novelty_vector=nv))
    cache = new_cache


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

    archive = []
    gen_idx = 0
    novel_prob = NOVEL_START
    elite = None
    while True:
        gen_idx += 1
        novel_prob = min(1.0, NOVEL_START + gen_idx * (1.0 - NOVEL_START) / NOVEL_GEN) 

        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            seeds, reward, steps, novelty_vector = output_queue.get()
            population.append((seeds, reward, novelty_vector))
            if np.random.uniform() > 0.01:
                archive.append(novelty_vector)
            batch_steps += steps
        if elite is not None:
            population.append(elite)

        for child in population:
            child = (child[0], child[1], compute_novelty_vs_archive(archive, child[2]))
        
        if np.random.uniform() < novel_prob:
            population.sort(key=lambda p: p[2], reverse=True)

        else:
            population.sort(key=lambda p: p[1], reverse=True)

        rewards = [p[1] for p in distances[:PARENTS_COUNT]]
        utils.write_ga(gen_idx, rewards, gen_seconds, batch_steps, speed, writer)
        print('%d: USING NOVELTY: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s' % (gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]

        for input_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
            input_queue.put(seeds)
        


if __name__ == '__main__':
    main()

