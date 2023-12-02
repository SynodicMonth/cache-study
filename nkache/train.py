from nkache.env import CacheEnv
from nkache.model import QNetwork, Transition, ReplayMemory

import torch
import torch.nn as nn

from rich import print
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

import random
import math

import argparse


class Agent:
    def __init__(self,
                 trace_file: str,
                 ckpt_dir: str,
                 num_sets: int = 2048,
                 associativity: int = 16,
                 block_size: int = 64,
                 hidden_dim: int = 128,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 eps_start: float = 0.9,
                 eps_end: float = 0.05,
                 eps_decay: int = 200,
                 tau: float = 0.01,
                 lr: float = 0.001) -> None:
        self.env = CacheEnv(num_sets, associativity, block_size)

        self.device = 'cpu'

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            
        # torch-directml was tried but it was slower than cpu :(

        print(f'using device: {self.device}')

        self.policy_net = QNetwork(self.env.observation_space(),
                                   self.env.action_space(), hidden_dim).to(self.device)
        self.target_net = QNetwork(self.env.observation_space(),
                                   self.env.action_space(), hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(10000)

        print(f'loading trace file: {trace_file}')
        self.env.load_trace(trace_file)
        print('preparing belady')
        self.env.prepare_belady()
        print('env ready')

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.lr)

        self.ckpt_dir = ckpt_dir

        self.steps_done = 0

    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + \
            (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space())]], device=self.device, dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # non-final mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1).values

        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        self.optimizer.step()

    def train(self, num_episodes=100, num_steps=50000):

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn(
                "[bold blue]{task.fields[hit_rate]:.5f} {task.fields[reward]}"),
            transient=True,
        )

        # show progress
        progress.start()

        state, done = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)

        for i_episode in range(num_episodes):
            curr_progress = progress.add_task(
                f"[cyan]Episode {i_episode+1}/{num_episodes}", total=num_steps, hit_rate=0, reward=0
            )
            total_reward = 0

            for t in range(num_steps):
                action = self._select_action(state)
                observation, reward, done = self.env.step(action.item())
                total_reward += reward

                reward = torch.tensor([reward], device=self.device)

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self._optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in target_net_state_dict:
                    target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + (
                        1 - self.tau) * target_net_state_dict[key]

                self.target_net.load_state_dict(target_net_state_dict)

                progress.update(curr_progress, advance=1, hit_rate=self.env.stats()[
                                'hit_rate'], reward=total_reward)

                if done:
                    hit_rate = self.env.stats()['hit_rate']
                    print(f'cache final hit rate: {hit_rate}')

                    state, done = self.env.reset()
                    state = torch.tensor(
                        state, device=self.device).unsqueeze(0)

            hit_rate = self.env.stats()['hit_rate']

            progress.update(curr_progress, completed=num_steps,
                            hit_rate=hit_rate, reward=total_reward, visible=False)

            print(
                f'episode {i_episode+1}/{num_episodes} hit rate: {hit_rate:.5f} reward: {total_reward}')

            # save model
            torch.save(self.policy_net.state_dict(),
                       f'{self.ckpt_dir}/policy_{i_episode+1}.pt')
            torch.save(self.target_net.state_dict(),
                       f'{self.ckpt_dir}/target_{i_episode+1}.pt')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--trace', type=str, required=True)
    argparser.add_argument('--ckpt-dir', type=str, required=True)
    argparser.add_argument('--num-sets', type=int, default=2048)
    argparser.add_argument('--associativity', type=int, default=16)
    argparser.add_argument('--block-size', type=int, default=64)
    argparser.add_argument('--hidden-dim', type=int, default=128)
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--gamma', type=float, default=0.99)
    argparser.add_argument('--eps-start', type=float, default=0.9)
    argparser.add_argument('--eps-end', type=float, default=0.05)
    argparser.add_argument('--eps-decay', type=int, default=200)
    argparser.add_argument('--tau', type=float, default=0.01)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--num-episodes', type=int, default=100)
    argparser.add_argument('--num-steps', type=int, default=50000)

    args = argparser.parse_args()

    agent = Agent(
        trace_file=args.trace,
        ckpt_dir=args.ckpt_dir,
        num_sets=args.num_sets,
        associativity=args.associativity,
        block_size=args.block_size,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        tau=args.tau,
        lr=args.lr
    )

    agent.train(num_episodes=args.num_episodes, num_steps=args.num_steps)


if __name__ == '__main__':
    main()
