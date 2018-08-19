#!/usr/bin/env python3
import copy
import gym
import numpy as np

from lib import model, utils


NOISE_STD = 0.01
POPULATION_SIZE = 50
PARENTS_COUNT = 10


def mutate_parent(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters:
        noise = np.random.normal(size=p.shape).astype(np.float32)
        p += NOISE_STD * noise
    return new_net


def main():
    env = gym.make('CartPole-v0')

    gen_idx = 0
    nets = [
        model.CartPoleNet(env.observation_space.shape[0], env.action_space.n) for _ in range(POPULATION_SIZE)
    ]
    population = [
        (net, evaluate(env, net)) for net in nets
    ]
    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)

        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f" % (gen_idx, reward_mean, reward_max, reward_std))
        if reward_mean > 199:
            print("Solved in %d steps" % gen_idx)
            break

        prev_population = population
        population = [population[0]]
        for _ in range(POPULATION_SIZE - 1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent)
            fitness = utils.evaluate(env, net)
            population.append((net, fitness))
        gen_idx += 1


if __name__ == '__main__':
    main()