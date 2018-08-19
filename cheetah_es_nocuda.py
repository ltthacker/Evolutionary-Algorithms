#!/usr/bin/env python3
import argparse
import collections
import gym
import time
import multiprocessing as mp
import numpy as np

from lib import model, optim


ENVIRONMENT = ''

NOISE_STD = 0.05
LEARNING_RATE = 0.01
PROCESSES_COUNT = 4
ITERS_PER_UPDATE = 10
MAX_ITERS = 100000


RewardsItem = collections.namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])


def make_env():
    return gym.make(ENVIRONMENT)


def train(optimizer, net, batch_noise, batch_reward, step_idx, noise_std):
    weighted_noise = None
    norm_reward = utils.compute_centered_ranks(np.array(batch_reward))

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    grads = []
    for p, p_update in zip(net.parameters, weighted_noise):
        update = p_update / (len(batch_reward) * noise_std)
        grads.append(-update)
    optimizer.step(grads)


def worker_func(worker_rd, params_queue, rewards_queue, noise_std):
    env = make_env()
    net = model.CheetahNet(env.observation_space.shape[0], env.action_space.shape[0])
    
    while True:
        params = params_queue.get()
        if params is None:
            break
        net.parameters = params

        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            pos_noise, neg_noise = sample_noise(net)
            pos_reward, pos_steps = utils.eval_with_noise(env, net, pos_noise, noise_std)
            neg_reward, neg_steps = utils.eval_with_noise(env, net, neg_noise, noise_std)
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward, neg_reward=neg_reward, steps=pos_steps+neg_steps))


def main():
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--iters', type=int, default=MAX_ITERS)
    args = parser.parse_args()

    env = make_env()
    net = model.CheetahNet(env.observation_space.shape[0], env.action_space.shape[0])

    params_queues = [mp.Queue(maxsize=1) for _ in range(PROCESSES_COUNT)]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    workers = []
    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker_func, args=(idx, params_queue, rewards_queue, device, args.noise_std))
        proc.start()
        workers.append(proc)

    print('All started')
    opt = optim.Adam(net.parameters, lr=args.lr)

    for step_idx in range(args.iters):
        params = net.parameters
        for q in params_queues:
            q.put(params)

        t_start = time.time()
        batch_noise = []
        batch_reward = []
        result = 0
        batch_steps = 0
        batch_steps_data = []
        while True:
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed)
                pos_noise, neg_noise = utils.sample_noise(net)
                batch_noise.append(pos_noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps
                batch_steps_data.append
                (reward.steps)
            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break
            time.sleep(0.01)

        dt_data = time.time() - t_start
        m_reward = np.mean(batch_reward)

        train(optimizer, net, batch_noise, batch_reward, writer, step_idx, args.noise_std)

        print("%d: reward=%.2f, speed=%.2f f/s, data_gather=%.3f, train=%.3f, steps_mean=%.2f, min=%.2f, max=%.2f, steps_std=%.2f" % (step_idx, m_reward, speed, dt_data, dt_step, np.mean(batch_steps_data), np.min(batch_steps_data), np.max(batch_steps_data), np.std(batch_steps_data)))
    
    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()


if __name__ == '__main__':
    main()