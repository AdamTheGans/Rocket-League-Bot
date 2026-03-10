# Project Overview: Rocket League Mechanic Curriculum Bot

### The Objective

The goal of this project is to train a state-of-the-art Rocket League AI capable of executing highly specific, complex mechanics (like the Kuxir pinch). Because these mechanics require microscopic precision and perfect timing, standard Reinforcement Learning (RL) from random states is highly inefficient.

To solve this, we have built a **custom Curriculum Learning framework** on top of RLGym v2 and `rlgym-ppo`. The bot starts by learning the mechanic in extremely easy, highly scripted setups. As it proves it can score, the environment dynamically increases the difficulty and adds physical noise until the bot can execute the mechanic from standard, chaotic game situations.

### Core Technology Stack

* **Environment Engine:** RLGym v2 & RocketSim (C++ physics simulator for blindingly fast environment stepping).
* **Algorithm:** Proximal Policy Optimization (PPO) via `rlgym-ppo`.
* **Deep Learning Backend:** PyTorch.

### What We Have Built So Far

You have successfully constructed a highly robust, multi-process training pipeline from scratch. Here are the core components currently functioning:

* **The "Oracle" Fast-Forward Condition:** Instead of waiting for the ball to slowly roll into the net after a pinch, we built a `FastForwardGoalCondition`. This spawns a "ghost" RocketSim arena in the background, fast-forwards the physics, and instantly terminates the episode if a goal is mathematically guaranteed. This saves massive amounts of compute time.
* **Dynamic Curriculum Architecture:** We implemented a `SharedCurriculum` manager that tracks the bot's success rate. If the bot's win rate crosses an 80% threshold, it automatically promotes the bot to a higher difficulty and increases the noise. If the bot fails too often, it demotes it.
* **Thread-Safe Multiprocessing Metrics:** Because RLGym uses multiple CPU workers in parallel to gather data, we engineered a completely thread-safe `multiprocessing.Queue` system. The individual worker environments drop their win/loss booleans into the queue, and the main PyTorch thread scoops them up at the end of every batch to calculate the true success rate.
* **RLGym v2 Compliance:** We successfully migrated complex custom logic (like `DoneConditions`, truncations, and multi-agent dictionary returns) from the legacy RLGym v1 API to the modern, lightning-fast v2 standard.

### Current Status: Live Training

As of right now, the infrastructure is complete. The environments are actively stepping, the C++ RocketSim physics are advancing, the multi-core workers are sending valid observation batches to the PyTorch neural networks, and the PPO algorithm is actively calculating gradients and updating the bot's "brain" (Actor/Critic networks). The training loop is fully operational.
