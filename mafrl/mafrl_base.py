import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from uav_mec_env import UAVMECEnv
from parameters import *
from dqn import DQN, ReplayMemory, Transition
import os
from common import CSVLogger

# Device setup
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class UE_Agent:
    def __init__(self, ue_id, n_observations, n_actions):
        self.ue_id = ue_id
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE_M)
        self.steps_done = 0

    def select_action(self, state, env, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        # Training mode: epsilon-greedy
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            )

        return action

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update_target(self):
        """Soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def set_parameters(self, state_dict):
        """Set the parameters of the agent's policy network."""
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_parameters(self):
        """Get the parameters of the agent's policy network."""
        return self.policy_net.state_dict()


class Server:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
    ):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)

    def reset(self):
        """Reset the server's policy network."""
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)

    def get_parameters(self):
        return self.policy_net.state_dict()

    def set_parameters(self, state_dict):
        self.policy_net.load_state_dict(state_dict)

    def federated_aggregate(self, agents_subset):
        """Aggregate parameters from a subset of agents."""
        total_params = {
            k: torch.zeros_like(v) for k, v in self.policy_net.state_dict().items()
        }
        for agent in agents_subset:
            agent_params = agent.get_parameters()
            for k in total_params.keys():
                total_params[k] += agent_params[k]

        # Average the parameters (only over the selected agents)
        for k in total_params.keys():
            total_params[k] /= len(agents_subset)

        self.set_parameters(total_params)


class MAFRL:
    def __init__(
        self,
        num_ues=100,
        num_uavs=10,
        num_episodes=1000,
        max_latency=1.0,
        fmax_uav=FMAX_UAV,
        task_data_size_min=TASK_DATA_MIN,
        task_data_size_max=TASK_DATA_MAX,
        task_cpu_cycles_min=TASK_CPU_MIN,
        task_cpu_cycles_max=TASK_CPU_MAX,
        outdir="results/",
    ):
        self.env = UAVMECEnv(
            num_ues=num_ues,
            num_uavs=num_uavs,
            episode_length=EPISODE_LENGTH,
            max_latency=max_latency,
            fmax_uav=fmax_uav,
            task_data_size_min=task_data_size_min,
            task_data_size_max=task_data_size_max,
            task_cpu_cycles_min=task_cpu_cycles_min,
            task_cpu_cycles_max=task_cpu_cycles_max,
        )
        self.num_ues = num_ues
        self.num_uavs = num_uavs
        self.num_episodes = num_episodes

        self.n_actions = self.env.total_actions
        self.n_observations = 2 + num_uavs * 2 + 3

        self.ue_agents = [
            UE_Agent(ue_id, self.n_observations, self.n_actions)
            for ue_id in range(self.num_ues)
        ]
        self.server = Server(self.n_observations, self.n_actions)

        self.aggregation_interval = 200
        self.target_update_interval = 1

        self.percent_selected = 0.2  # Percentage of agents selected for aggregation
        self.min_agents_train = (
            20  # Minimum number of agents to participate in training
        )

        self.num_rounds = 400  # Number of rounds for training

        self.step_done = 0

        self.csv_logger = CSVLogger(
            os.path.join(outdir, f"mafrl_new_training_log.csv"),
            fieldnames=[
                "episode",
                "policy",
                "reward",
                "total_power_system",
                "total_power_ue",
                "total_power_uav",
                "violations",
                "vio_local",
                "vio_offload",
            ],
        )

    def train_and_update(self, ue_agent):
        """Train the UE agent and update its target network."""
        ue_agent.optimize_model()
        ue_agent.soft_update_target()

    def train(self):
        self.env.reset()
        cur_episode = 0
        for rnd in range(self.num_rounds):
            # num_agents_selected = max(
            #     int(self.percent_selected * self.num_ues), self.min_agents_train
            # )
            # selected_agents = random.sample(self.ue_agents, num_agents_selected)

            # Initial aggregation
            for ue_agent in self.ue_agents:
                ue_agent.set_parameters(self.server.get_parameters())

            # Training
            states = []
            for ue_agent in self.ue_agents:
                state = self.env.get_state_ue(ue_agent.ue_id)
                state = torch.tensor(
                    state, device=device, dtype=torch.float32
                ).unsqueeze(0)
                states.append(state)

            for episode in range(int(self.num_episodes / self.num_rounds)):
                actions = []
                for ue_agent, state in zip(self.ue_agents, states):
                    action = ue_agent.select_action(state, self.env)
                    actions.append(action)

                self.step_done += 1

                _, reward, terminateds, _, info = self.env.step(actions)
                if cur_episode % 10 == 0:
                    print(
                        f"Episode {episode}/{self.num_episodes} - Total Power: {info['total_power_system']:.3f} W, Reward: {reward:.3f}"
                    )

                next_states = []
                reward_tensor = torch.tensor(
                    reward, device=device, dtype=torch.float32
                ).unsqueeze(0)

                for ue_agent in self.ue_agents:
                    next_state = torch.tensor(
                        self.env.get_state_ue(ue_agent.ue_id),
                        device=device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    next_states.append(next_state)

                for ue_agent in self.ue_agents:
                    ue_agent.memory.push(
                        states[ue_agent.ue_id],
                        actions[ue_agent.ue_id],
                        next_states[ue_agent.ue_id],
                        reward_tensor,
                    )
                states = next_states

                for ue_agent in self.ue_agents:
                    ue_agent.optimize_model()
                    ue_agent.soft_update_target()

                # Partial Aggregation
                if (cur_episode + 1) % self.aggregation_interval == 0:

                    self.server.federated_aggregate(self.ue_agents)
                    for ue_agent in self.ue_agents:
                        ue_agent.set_parameters(self.server.get_parameters())
                        ue_agent.steps_done = self.step_done

                cur_episode += 1
                self.csv_logger.log(
                    {
                        "episode": cur_episode,
                        "policy": "MAFRL",
                        "reward": round(reward, 6),
                        "total_power_system": round(info["total_power_system"], 6),
                        "total_power_ue": round(info["total_power_ue"], 6),
                        "total_power_uav": round(info["total_power_uav"], 6),
                        "violations": round(info["violations"], 6),
                        "vio_local": round(info["vio_local"], 6),
                        "vio_offload": round(info["vio_offload"], 6),
                    }
                )

    def save_model(self, filename):
        """Save the model parameters to a file."""
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.server.get_parameters(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load the model parameters from a file."""
        if os.path.exists(filename):
            state_dict = torch.load(filename, map_location=device)
            self.server.set_parameters(state_dict)
            for ue_agent in self.ue_agents:
                ue_agent.set_parameters(state_dict)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} does not exist.")

    def test(self, num_steps=200):
        """Test the trained model."""
        print("Testing started...")
        avg_power_system = 0.0
        avg_power_ue = 0.0
        avg_power_uav = 0.0
        avg_violations = 0.0
        avg_vio_local = 0.0
        avg_vio_offload = 0.0
        avg_reward = 0.0

        states = []
        for ue_id in range(self.env.num_ues):
            state = torch.tensor(
                self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
            ).unsqueeze(0)
            states.append(state)

        for step in range(num_steps):
            actions = []
            for ue_id in range(self.env.num_ues):
                state = states[ue_id]
                action = self.ue_agents[ue_id].select_action(
                    state, self.env, eval_mode=True
                )
                actions.append(action)

            _, reward, terminateds, _, info = self.env.step(actions)

            next_states = []
            for ue_id in range(self.env.num_ues):
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
                ).unsqueeze(0)
                next_states.append(next_state)

            states = next_states

            avg_power_system += info["total_power_system"]
            avg_power_ue += info["total_power_ue"]
            avg_power_uav += info["total_power_uav"]
            avg_violations += info["violations"]
            avg_vio_local += info["vio_local"]
            avg_vio_offload += info["vio_offload"]
            avg_reward += reward

        return {
            "avg_power_system": avg_power_system / num_steps,
            "avg_power_ue": avg_power_ue / num_steps,
            "avg_power_uav": avg_power_uav / num_steps,
            "avg_violations": avg_violations / num_steps,
            "avg_vio_local": avg_vio_local / num_steps,
            "avg_vio_offload": avg_vio_offload / num_steps,
            "avg_reward": avg_reward / num_steps,
        }
