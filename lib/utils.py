import numpy as np

def sample_noise(net):
    pos = []
    neg = []
    for p in net.parameters:
        noise = np.random.normal(size=p.shape).astype(np.float32)
        pos.append(noise)
        neg.append(-noise)
    return pos, neg


def states_preprocessor(states):
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return np.states.astype(np.float32)


def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    steps = 0
    done = False
    while not done:
        obs = states_preprocessor(obs)
        actions = net(obs)
        if net.activation == 'softmax':
            actions = np.argmax(actions, axis=1)
        elif net.activation == 'tanh':
            pass
        obs, r, done, _ = env.step(actions[0])
        reward += r
        steps += 1
    return reward, steps


def eval_with_noise(env, net, noise, noise_std):
    old_params = net.parameters
    for p, p_n in zip(net.parameters, noise):
        p += noise_std * p_n
    r, s = evaluate(env, net)
    net.parameters = old_params
    return r, s


def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def write_ga(gen_idx, rewards, gen_seconds, batch_steps, speed, writer):
    reward_mean = np.mean(rewards)
    reward_max = np.max(rewards)
    reward_std = np.std(rewards)

    writer.add_scalar("reward_mean", reward_mean, gen_idx)
    writer.add_scalar("reward_std", reward_std, gen_idx)
    writer.add_scalar("reward_max", reward_max, gen_idx)
    writer.add_scalar("batch_steps", batch_steps, gen_idx)
    writer.add_scalar("gen_seconds", gen_seconds, gen_idx)
    writer.add_scalar("speed", speed, gen_idx)
