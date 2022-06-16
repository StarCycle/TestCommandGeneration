import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":

    learning_rate = 2.5e-4               # Initial learning rate
    num_env_steps = 512                  # How many steps you interact with the env in a round 
    num_update_steps = 4                 # How many steps you update the neural networks in a round
    minibatch_size = 32                  # The batch size to update the neural networks
    total_timesteps = 50000              # How many steps you interact with the env
    gamma = 0.99                         # Decay rate of future rewards
    gae_lambda = 0.95                    # Parameter in advantage estimation
    clip_coef = 0.2                      # Parameter to clip the (p_new/p_old) ratio
    ent_coef = 0.01                      # Weight of the entropy loss in the total loss
    vf_coef = 0.5                        # Weight of the value loss in the total loss
    max_grad_norm = 0.5                  # max norm of the gradient vector

    writer = SummaryWriter("runs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: seeding
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize storage
    obs = torch.zeros((num_env_steps, env.observation_space.shape[0])).to(device)
    actions = torch.zeros(num_env_steps).to(device)
    logprobs = torch.zeros(num_env_steps).to(device)
    rewards = torch.zeros(num_env_steps).to(device)
    dones = torch.zeros(num_env_steps).to(device)
    values = torch.zeros(num_env_steps).to(device)
    next_obs = torch.Tensor(env.reset()).to(device)
    next_done = torch.zeros(1).to(device)

    num_updates = total_timesteps // num_env_steps
    global_step = 0
    for update in range(1, num_updates + 1):

        # Annealing the learning rate.
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_env_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            if done == 1:
                next_obs = env.reset()
            rewards[step] = torch.tensor(reward).to(device).view(-1) 
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0 
            for t in reversed(range(num_env_steps)):
                if t == num_env_steps - 1:
                    nextnonterminal = 1.0 - next_done 
                else:
                    nextnonterminal = 1.0 - dones[t + 1] 
                    next_value = values[t + 1]
                delta = rewards[t] + gamma * next_value * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Optimizing the policy and value network
        inds = np.arange(num_env_steps) 
        clipfracs = []
        for epoch in range(num_update_steps):
            np.random.shuffle(inds)
            for start in range(0, num_env_steps, minibatch_size):
                end = start + minibatch_size
                minds = inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[minds], actions.long()[minds])
                logratio = newlogprob - logprobs[minds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                madvantages = advantages[minds]

                # Policy loss
                pg_loss1 = -madvantages * ratio
                pg_loss2 = -madvantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) 
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns[minds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # Update the neural networks
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("dones_sum", dones.sum().item(), global_step)
        print("step:", global_step, "dones_sum", dones.sum().item())

    env.close()
    writer.close()
