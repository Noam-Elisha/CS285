from agents.algorithm.ppo    import PPO
from agents.algorithm.sac    import SAC
from agents.agent            import Agent

from discriminators.gail     import GAIL
from discriminators.vail     import VAIL
from discriminators.airl     import AIRL
from discriminators.vairl    import VAIRL
from discriminators.eairl    import EAIRL
from discriminators.sqil     import SQIL
from utils.utils             import RunningMeanStd, Dict, make_transition

from configparser            import ConfigParser
from argparse                import ArgumentParser

import os
import gym
import numpy as np
import time 
import torch

os.makedirs('./model_weights', exist_ok=True)

env = gym.make("Hopper-v2")
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

parser = ArgumentParser('parameters')
timestr = time.strftime("%Y%m%d-%H%M%S")

parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'ppo', help = 'actor training algorithm(default: ppo)')
parser.add_argument("--discriminator", type=str, default = 'gail', help = 'discriminator training algorithm(default: gail)')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: True)')
parser.add_argument('--transformer', type=bool, default=False, help='use transformer architechture')

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

demonstrations_location_args = Dict(parser,'demonstrations_location',True)
agent_args = Dict(parser,args.agent)
discriminator_args = Dict(parser,args.discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = f'./runs/{args.agent}_{args.discriminator}_T{args.transformer}_{timestr}')
else:
    writer = None

if args.discriminator == 'airl':
    discriminator = AIRL(writer, device, state_dim, action_dim, discriminator_args, args.transformer)
elif args.discriminator == 'vairl':
    discriminator = VAIRL(writer, device, state_dim, action_dim, discriminator_args, args.transformer)
elif args.discriminator == 'gail':
    discriminator = GAIL(writer, device, state_dim, action_dim, discriminator_args, args.transformer)
elif args.discriminator == 'vail':
    discriminator = VAIL(writer,device,state_dim, action_dim, discriminator_args, args.transformer)
elif args.discriminator == 'eairl':
    discriminator = EAIRL(writer, device, state_dim, action_dim, discriminator_args, args.transformer)
elif args.discriminator == 'sqil':
    discriminator = SQIL(writer, device, state_dim, action_dim, discriminator_args, args.transformer)
else:
    raise NotImplementedError
    
if args.agent == 'ppo':
    algorithm = PPO(device, state_dim, action_dim, agent_args)
elif args.agent == 'sac':
    algorithm = SAC(device, state_dim, action_dim, agent_args)
else:
    raise NotImplementedError
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
    
import shutil, glob

shutil.copyfile('config.ini', f'./runs/{args.agent}_{args.discriminator}_T{args.transformer}_{timestr}/config.ini')
state_rms = RunningMeanStd(state_dim)

score_lst = []
discriminator_score_lst = []
score = 0.0
discriminator_score = 0
stoppp
if agent_args.on_policy == True:
    state_lst = []
    state_ = (env.reset())
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    # timeline = np.expand_dims(state, 0)
    for n_epi in range(args.epochs):
        timeline = torch.zeros(0)
        actions = torch.zeros(0)
        timesteps = torch.zeros(0)
        next_states = torch.zeros(0)
        for t in range(agent_args.traj_length):
            timesteps = torch.cat((timesteps, torch.tensor([[t]])), -1)
            timeline = torch.cat((timeline, torch.tensor(state).unsqueeze(0)))
            
            if args.render:    
                env.render()
            state_lst.append(state_)
            action, log_prob = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).to(device))
        
            actions = torch.cat((actions.to(device=device), action.unsqueeze(0).to(device=device)), 1)
            next_state_, r, done, info = env.step(action.squeeze().cpu().numpy())
            
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            
            next_states = torch.cat((next_states, torch.tensor(next_state).unsqueeze(0)))
            if discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                                        log_prob,\
                                        torch.tensor(timeline).unsqueeze(0).float().to(device),actions,\
                                        torch.tensor(next_states).unsqueeze(0).float().to(device),\
                                                              torch.tensor(done).view(1,1)\
                                                 )[:,-1]
                # print(reward.size())
            else:
                reward = discriminator.get_reward(torch.tensor(timeline).unsqueeze(0).float().to(device),actions)[:,-1]

            transition = make_transition(state,\
                                         action.cpu(),\
                                         np.array([reward.cpu()/10.0]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += r
            discriminator_score += reward
            if done or (t==agent_args.traj_length):
                state_ = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if writer != None:
                    writer.add_scalar("score/real", score, n_epi)
                    writer.add_scalar("score/discriminator", discriminator_score, n_epi)
                score = 0
                discriminator_score = 0
            else:
                state = next_state
                # timeline = np.concatenate((timeline, np.expand_dims(next_state, 0)))
                state_ = next_state_
        agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi)
        state_rms.update(np.vstack(state))
        state_lst = []
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if (n_epi % args.save_interval == 0 )& (n_epi != 0):
            torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))
else: #off-policy
    for n_epi in range(args.epochs):
        score = 0.0
        discriminator_score = 0.0
        state = env.reset()
        # timeline = np.expand_dims(state, 0)
        done = False
        timeline = torch.zeros(0)
        actions = torch.zeros(0)
        # timesteps = torch.zeros(0)
        while not done:
            if args.render:    
                env.render()
            
            # timesteps = torch.cat((timesteps, torch.tensor([[t]])), -1)
            timeline = torch.cat((timeline, torch.tensor(state).unsqueeze(0)))
            action_, log_prob = agent.get_action(torch.from_numpy(state).float().to(device))
            
            actions = torch.cat((actions.to(device=device), action_.unsqueeze(0).to(device=device)), 1)


            action = action_.cpu().detach().numpy()
            # print(action.shape, env.action_space.sample().shape)
            next_state, r, done, info = env.step(action[0])
            if discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                            log_prob,
                            torch.tensor(timeline).unsqueeze(0).float().to(device),actions,\
                            torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                  torch.tensor(done).unsqueeze(0)\
                                                 )[:,-1]
            else:
                reward = discriminator.get_reward(torch.tensor(timeline).unsqueeze(0).float().to(device),actions)[:,-1]

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward.cpu()/10.0]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 

            # timeline = np.concatenate((timeline, np.expand_dims(next_state, 0)))
            state = next_state

            score += r
            discriminator_score += reward
            
            if agent.data.data_idx > agent_args.learn_start_size: 
                agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi, agent_args.batch_size)
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
            writer.add_scalar("score/discriminator", discriminator_score, n_epi)
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
            
