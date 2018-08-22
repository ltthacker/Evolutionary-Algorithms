# Evolutionary-Algorithms
Evolutionary strategies and genetic algorithms for CartPole and HalfCheetah

I implemented evolutionary algorithm and genetic strategy in NumPy for Gym's CartPole environment.  I used Roboschool as an alternative to Mujoco for genetic strategies in HalfCheetah, and used PyTorch for multiprocessing and CUDA.
Using Maxim Lapan's base code for genetic strategy,[^1] I added a few new implementations following Uber AI's paper *Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning*.[^2]  

* In [cheetah_ga.py](https://github.com/rhshi/Evolutionary-Algorithms/blob/master/cheetah_ga.py), I added a more robust method of obtaining the elite offspring in each generation using additional episodes and recording the highest mean reward.
* In [cheeta_ga_ns.py](https://github.com/rhshi/Evolutionary-Algorithms/blob/master/cheetah_ga_ns.py), I implemented novelty search.  The purpose of this, especially in movement based environments like HalfCheetah, is to search for novel actions as opposed to maximizing the objective (which is to walk).  By tuning some of the hyperparameters at the top of the script (NOVEL_START and NOVEL_GEN), one can switch from pure fitness maximization to pure novelty search.  The default is to start the probabilty that the agent searches for novel actions at 0.05 and then is annealed to 1 after 200 frames; 200 frames was chosen because that is when the agent begins to perfect standing up at around a reward of 1000, but is unable to begin walking.  Novelty search is aimed at allowing the agent to discover walking.

[^1]: [https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
[^2]: [https://arxiv.org/pdf/1712.06567.pdf](https://arxiv.org/pdf/1712.06567.pdf)
