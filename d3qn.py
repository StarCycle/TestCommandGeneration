import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from GNN_Agent_sum import GNN_Agent
from MyEnv import MyEnv

class ReplayBuffer(object):
    def __init__(self, capacity, observation_len, device):
        self.obs = torch.zeros(capacity, observation_len).to(device)
        self.actions = torch.zeros(capacity, dtype=torch.long).to(device)
        self.next_obs = torch.zeros(capacity, observation_len).to(device)
        self.rewards = torch.zeros(capacity).to(device)
        self.dones = torch.zeros(capacity).to(device)
        self.counter = 0
        self.capacity = capacity

    def store(self, observation, action, next_observation, reward, done):
        index = self.counter % self.capacity
        self.obs[index, :] = observation
        self.actions[index] = action
        self.next_obs[index, :] = next_observation
        self.rewards[index] = reward
        self.dones[index] = done
        self.counter += 1

    def sample(self, batch_size):
        if self.counter > self.capacity:
            index = np.random.choice(self.capacity, size=batch_size)
        else:
            index = np.random.choice(self.counter, size=batch_size)
        return self.obs[index, :], self.actions[index], self.next_obs[index, :], self.rewards[index], self.dones[index]

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def train(env, name, buffer_size, batch_size, learning_rate, exploration_fraction, learning_starts, train_frequency, gamma, target_network_frequency, total_timesteps, out_channels, gnn_layers):

    start_e = 1
    end_e = 0.01

    # seeding
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter("runs/"+name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = GNN_Agent(env, out_channels, gnn_layers, device).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = GNN_Agent(env, out_channels, gnn_layers, device).to(device)
    target_network.load_state_dict(q_network.state_dict())
    rb = ReplayBuffer(buffer_size, len(env.recordCov+env.actions), device)
    
    cumulative_reward = 0
    episode_return = 0
    # start the game
    observation = torch.tensor(env.reset()).to(device)
    for global_step in range(total_timesteps):
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            action = random.randint(0, len(env.actions) - 1)
        else:
            logits = q_network(observation)
            action = torch.argmax(logits).cpu().numpy()

        # execute the game and log data.
        next_ob, reward, done, info = env.step(action)
        cumulative_reward += reward
        episode_return = reward + gamma*episode_return
        print("global step:", global_step, "cumulative rewards:", cumulative_reward, 'covsum', env.covSum)
        if done == True:
            writer.add_scalar("cumulative_reward", cumulative_reward, global_step)
            writer.add_scalar("episode_return", episode_return, global_step)
            writer.add_scalar("cov sum", env.covSum, global_step)
            cumulative_reward = 0
            next_ob = env.reset()
        next_ob = torch.tensor(next_ob, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward).to(device)
        done = torch.tensor(done).to(device)

        # save data to reply buffer
        real_next_ob = next_ob.clone()
        rb.store(observation, action, real_next_ob, reward, done)

        # step easy to overlook
        observation = next_ob

        # training
        if global_step > learning_starts and global_step % train_frequency== 0:
            sampled_obs, sampled_actions, sampled_next_obs, sampled_rewards, sampled_dones = rb.sample(batch_size)
            with torch.no_grad():
                td_actions = q_network(sampled_next_obs).argmax(dim=1)
                td_actions = td_actions.reshape((len(sampled_actions), -1))
                target_max = target_network(sampled_next_obs).gather(1, td_actions).squeeze()
                td_target = sampled_rewards + gamma * target_max * (1 - sampled_dones)
            sampled_actions = sampled_actions.reshape((len(sampled_actions), -1))
            old_val = q_network(sampled_obs).gather(1, sampled_actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            # save the network
            if global_step % 10 == 0:
                torch.save(q_network.state_dict(), 'Q_Network_'+name)

    writer.close()

if __name__ == "__main__":

    buffer_size = [500000]
    batch_size = [128]
    learning_rate = [1e-4]
    exploration_fraction = [0.9]
    learning_starts = [128]
    train_frequency = [16]
    gamma = [0.9]
    target_network_frequency = [500]
    total_timesteps = [500000]
    out_channels = [128]
    gnn_layers = [5]
    num_epoch_steps = 128

    env = MyEnv('COMMS', 4, 'para.csv', 'telec.csv', 'telem.csv', num_epoch_steps, 630)

    for bufs in buffer_size:
        for bs in batch_size:
            for lr in learning_rate:
                for ef in exploration_fraction:
                    for ls in learning_starts:
                        for tf in train_frequency:
                            for ga in gamma:
                                for tnf in target_network_frequency:
                                    for tt in total_timesteps:
                                        for oc in out_channels:
                                            for gl in gnn_layers:
                                                name = 'bufs'+str(bufs)+'_bs'+str(bs)+'_lr'+str(lr)+'_ef'+str(ef)+'_ls'+str(ls)+'_tf'+str(tf)+'_ga'+str(ga)+'_tnf'+str(tnf)+'_tt'+str(tt)+'_oc'+str(oc)+'_gl'+str(gl)
                                                train(env, name, bufs, bs, lr, ef, ls, tf, ga, tnf, tt, oc, gl)