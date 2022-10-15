import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from GNN_Agent_sum import GNN_Agent
from MyEnv import MyEnv

def train(env, name, target_kl, minibatch_size, gamma, ent_coef, vf_coef, learning_rate, total_timesteps, gnn_layers, num_epoch_steps):

    num_env_steps = 128                  # How many steps you interact with the env before an update
    num_update_steps = 4                 # How many times you update the neural networks after interation
    gae_lambda = 0.95                    # Parameter in advantage estimation
    clip_coef = 0.2                      # Parameter to clip the (p_new/p_old) ratio
    max_grad_norm = 0.5                  # max norm of the gradient vector

    writer = SummaryWriter('runs/' + name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = GNN_Agent(env, 'graph_len127.json', gnn_layers, device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: seeding
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize storage for a round
    obs = torch.zeros((num_env_steps, len(env.recordCov+env.actions))).to(device) # My environment
    # obs = torch.zeros((num_env_steps, env.observation_space.shape[0])).to(device) # gym
    actions = torch.zeros(num_env_steps).to(device)
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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = env.step(action.item()) 
            cumu_rewards += reward
            print("global step:", global_step, "reward", reward, "average clc per loop:", cumu_rewards/num_epoch_steps*1000)
            if done == 1:
                writer.add_scalar("cumulative reward", cumu_rewards, global_step)
                writer.add_scalar("average clc per loop", cumu_rewards/num_epoch_steps*1000, global_step)
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
            approx_kl = []
            np.random.shuffle(inds)
            for start in range(0, num_env_steps, minibatch_size):
                end = start + minibatch_size
                minds = inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[minds], actions.long()[minds])
                logratio = newlogprob - logprobs[minds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl += [((ratio - 1) - logratio).mean().item()]
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                # Policy loss
                pg_loss1 = -advantages[minds] * ratio
                pg_loss2 = -advantages[minds] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) 
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
            if np.mean(approx_kl) > target_kl:
                optimizer.param_groups[0]["lr"] *= 0.99
		
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("value_loss", v_loss.item(), global_step)
        writer.add_scalar("policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("entropy", entropy_loss.item(), global_step)
        writer.add_scalar("approx_kl", np.mean(approx_kl), global_step)
        writer.add_scalar("clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("explained_variance", explained_var, global_step)
        writer.add_scalar("mean_value", values.mean().item(), global_step)
        torch.save(agent.state_dict(), 'PPO_'+name)

    writer.close()

if __name__ == "__main__":

    target_kl = [0.02]	
    minibatch_size = [32]	
    gamma = [0.9]
    ent_coef = [0.01]	            
    vf_coef = [0.5]	
    total_timesteps = [100000]	
    learning_rate = [1e-4]
    gnn_layers = [5]
    num_epoch_steps = 128

    env = MyEnv('COMMS', 4, 'para.csv', 'telec.csv', 'telem.csv', num_epoch_steps, 630)

    for tk in target_kl:
        for bs in minibatch_size:
            for ga in gamma:
                for ef in ent_coef:
                    for vf in vf_coef:
                        for lr in learning_rate:
                            for tt in total_timesteps:
                                for gl in gnn_layers:
                                    name = 'tk'+str(tk)+'_bs'+str(bs)+'_ga'+str(ga)+'_ef'+str(ef)+'_vf'+str(vf)+'_lr'+str(lr)+'_tt'+str(tt)+'_gl'+str(gl)
                                    train(env, name, tk, bs, ga, ef, vf, lr, tt, gl, num_epoch_steps)