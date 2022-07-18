import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from MyEnv import MyEnv

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env, num_nn, critic_std, actor_std):
        super().__init__()
        self.action_space = env.action_space
        self.critic = nn.Sequential(
            layer_init(nn.Linear(len(env.recordCov), num_nn)), # My environment
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, num_nn)),
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, 1), std=critic_std),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(len(env.recordCov), num_nn)), # My environment
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, num_nn)),
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, sum(self.action_space)), std=actor_std), # My environment
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        split_logits = torch.split(logits, self.action_space, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.critic(x)

def train(env, name, target_kl, minibatch_size, gamma, ent_coef, vf_coef, num_nn, critic_std, actor_std, num_epoch_steps):

    learning_rate = 5e-4
    total_timesteps = 200000     # How many steps you interact with the env
    num_env_steps = 128         # How many steps you interact with the env before an update
    num_update_steps = 4       # How many times you update the neural networks after interation
    gae_lambda = 0.95             # Parameter in advantage estimation
    clip_coef = 0.2                    # Parameter to clip the (p_new/p_old) ratio
    max_grad_norm = 0.5         # max norm of the gradient vector

    writer = SummaryWriter('runs/' + name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, num_nn, critic_std, actor_std).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: seeding
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize storage for a round
    obs = torch.zeros((num_env_steps, len(env.recordCov))).to(device) # My environment
    actions = torch.zeros(num_env_steps, len(env.action_space)).to(device)
    logprobs = torch.zeros(num_env_steps).to(device)
    rewards = torch.zeros(num_env_steps).to(device)
    dones = torch.zeros(num_env_steps).to(device)
    values = torch.zeros(num_env_steps).to(device)
    next_obs = torch.Tensor(env.reset()).to(device)
    next_done = torch.zeros(1).to(device)

    global_step = 0
    cumu_rewards = 0
    num_rounds = total_timesteps // num_env_steps
    for round in range(num_rounds):
            
        # ALGO LOGIC: action logic
        for step in range(num_env_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
                            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.reshape((1, -1)))
                action = action.reshape(-1)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = env.step(action.cpu().numpy()) 
            cumu_rewards += reward
            print("global step:", global_step, "cumulative rewards:", cumu_rewards, 'covsum', env.covSum)
            if done == True:
                writer.add_scalar("cumulative rewards", cumu_rewards, global_step)
                writer.add_scalar("cov sum", env.covSum, global_step)
                next_obs = env.reset()
                cumu_rewards = 0
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
        for update in range(num_update_steps):
            np.random.shuffle(inds)
            for start in range(0, num_env_steps, minibatch_size):
                end = start + minibatch_size
                minds = inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[minds], actions.long()[minds].T)
                logratio = newlogprob - logprobs[minds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                # Policy loss
                pg_loss1 = - advantages[minds] * ratio
                pg_loss2 = - advantages[minds] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) 
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
		
            # Annealing the learning rate, if KL is too high
            if approx_kl > target_kl:
                optimizer.param_groups[0]["lr"] *= 0.99
		
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("value_loss", v_loss.item(), global_step)
        writer.add_scalar("policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("entropy", entropy_loss.item(), global_step)
        writer.add_scalar("approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("explained_variance", explained_var, global_step)
        writer.add_scalar("mean_value", values.mean().item(), global_step)
        for name, param in agent.named_parameters():
            writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=global_step)
            writer.add_histogram(tag=name+'_data', values=param.data, global_step=global_step)

    torch.save(agent.state_dict(), 'Model_'+name)
    writer.close()

if __name__ == "__main__":

    target_kl = [0.02, 0.1]	# Max KL divergence
    minibatch_size = [32]   # The batch size to update the neural network
    gamma = [0.9, 0.95]
    ent_coef = [0.01, 0.05]	# Weight of the entropy loss in the total loss
    vf_coef = [0.5]	        # Weight of the value loss in the total loss
    num_nn = [1024, 2048]
    critic_std = [1]
    actor_std = [0.01]
    num_epoch_steps = 128    # How many steps you interact with the env before a reset

    env = MyEnv('COMMS', 4, 'para.csv', [3, 17, 18, 25], 3, num_epoch_steps, 631)

    for tk in target_kl:
        for bs in minibatch_size:
            for ga in gamma:
                for ef in ent_coef:
                    for vf in vf_coef:
                        for num in num_nn:
                            for cstd in critic_std:
                                for astd in actor_std:
                                    name = 'tk'+str(tk)+'_bs'+str(bs)+'_ga'+str(ga)+'_ef'+str(ef)+'_vf'+str(vf)+'_num'+str(num)+'_cs'+str(cstd)+'_as'+str(astd)
                                    train(env, name, tk, bs, ga, ef, vf, num, cstd, astd, num_epoch_steps)
