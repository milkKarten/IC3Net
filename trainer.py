from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import time

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask',
                                       'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env, multi=False):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        # print("1trainer", getargspec(self.env.reset).args, self.env.reset)
        self.display = False
        self.last_step = False
        if self.args.optim_name == "RMSprop":
            self.optimizer = optim.RMSprop(policy_net.parameters(),
                lr = args.lrate, alpha=0.97, eps=1e-6)
        elif self.args.optim_name == "Adadelta":
            self.optimizer = optim.Adadelta(policy_net.parameters())#, lr = args.lrate)
        if self.args.scheduleLR:
            self.load_scheduler(start_epoch=0)
        self.params = [p for p in self.policy_net.parameters()]
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')
        if multi:
            self.device = torch.device('cpu')
        print("Device:", self.device)
        self.success_metric = 0
        self.epoch_success = 0
        self.cur_epoch_i = 0

        # traffic junction success
        if self.args.env_name == "traffic_junction":
            if self.args.difficulty == 'easy':
                self.success_thresh = .90
                # self.success_thresh = .95
            elif self.args.difficulty == 'medium':
                self.success_thresh = .86
                # self.success_thresh = .9
            elif self.args.difficulty == 'hard':
                self.success_thresh = .70
        else:
            self.success_thresh = 1.0


        # reward communication when false
        self.args.gating_punish = True

        self.reward_epoch_success = 0
        self.reward_success = 0
        self.cur_reward_epoch_i = 0

        # reward tuning
        self.last_error = None
        self.total_error = None

        # traffic junction curriculum
        self.begin_tj_curric = False
        self.tj_epoch_success = 0
        self.tj_success = 0
        self.tj_epoch_i = 0

        # communication curriculum with hard constraint
        self.min_budget = 0.05
        self.policy_net.budget = self.args.budget
        self.end_comm_curric = True
        self.comm_epoch_i = 0
        self.comm_epoch_success = 0
        self.comm_success = 0

        # if comunication has converged at budget
        self.comm_converge = False
        # self.comm_scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.01)
        self.loss_autoencoder = None
        self.loss_min_comm = None

    def success_curriculum(self, success_rate, num_episodes):
        if self.args.variable_gate:
            self.cur_epoch_i += 1
            self.epoch_success += success_rate
            # print("cur i", self.cur_epoch_i, self.success_metric, num_episodes, success_rate)
            if self.cur_epoch_i >= self.args.epoch_size:
                self.cur_epoch_i = 0
                if self.epoch_success / float(num_episodes*self.args.epoch_size) > self.success_thresh:
                    # print(self.epoch_success / float(num_episodes*self.args.epoch_size), self.success_thresh)
                    self.success_metric += 1
                else:
                    self.success_metric = 0
                self.epoch_success = 0

            # print("success curriculum", self.success_metric / max(1, self.args.nprocesses))
            if self.success_metric  >= 20: #/ max(1, self.args.nprocesses) >= 20:
                self.args.comm_action_one = False
                self.args.variable_gate = False

    def reward_curriculum(self, success_rate, num_episodes):
        if self.args.gate_reward_curriculum and not self.args.variable_gate:
            self.cur_reward_epoch_i += 1
            self.reward_epoch_success += success_rate
            if self.cur_reward_epoch_i >= self.args.epoch_size:
                self.cur_reward_epoch_i = 0
                if self.reward_epoch_success / float(num_episodes*self.args.epoch_size) > self.success_thresh:
                    self.reward_success += 1
                else:
                    self.reward_success = 0
                self.reward_epoch_success = 0
            if self.reward_success >= 20:
                self.args.gating_punish = True
                self.args.gate_reward_curriculum = False
                # reset optimizer
                self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                    lr = args.lrate, alpha=0.97, eps=1e-6)

    def tj_curriculum(self, success_rate, num_episodes):
        if not self.begin_tj_curric and False:
            self.tj_epoch_i += 1
            self.tj_epoch_success += success_rate
            if self.tj_epoch_i >= self.args.epoch_size:
                self.tj_epoch_i = 0
                if self.tj_epoch_success / float(num_episodes*self.args.epoch_size) > self.success_thresh:
                    self.tj_success += 1
                else:
                    self.tj_success = 0
                self.tj_epoch_success = 0
            if self.tj_success >= 20:
                self.begin_tj_curric = True

    def communication_curriculum(self, success_rate, num_episodes):
        if not self.end_comm_curric and not self.args.variable_gate:
            self.comm_epoch_i += 1
            self.comm_epoch_success += success_rate
            if self.comm_epoch_i >= self.args.epoch_size:
                self.comm_epoch_i = 0
                if self.comm_epoch_success / float(num_episodes*self.args.epoch_size) > self.success_thresh:
                    self.comm_success += 1
                else:
                    self.comm_success = 0
                self.comm_epoch_success = 0
            if self.comm_success >= 50:
                # decrease budget, reset curriculum params
                self.policy_net.budget -= 0.05
                self.comm_epoch_i = 0
                self.comm_epoch_success = 0
                self.comm_success = 0
            if self.policy_net.budget <= self.min_budget:
                self.end_comm_curric = True

    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        # print(reset_args, " trainer", self.env.reset)
        if 'epoch' in reset_args:
            state = self.env.reset(epoch, success=self.begin_tj_curric)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step
        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        # one is used because of the batch size.
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        if self.args.ic3net:
            stat['budget'] = self.policy_net.budget

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                info['comm_budget'] = np.zeros(self.args.nagents, dtype=int)
                info['step_t'] = t  # episode step for resetting communication budget
                stat['comm_action'] = np.zeros(self.args.nagents, dtype=int)[:self.args.nfriendly]

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, info)
                if self.args.autoencoder and not self.args.autoencoder_action:
                    decoded = self.policy_net.decode()
                    x_all = x[0].sum(dim=1).expand(self.args.nagents, -1).reshape(decoded.shape)
                    if self.loss_autoencoder == None:
                        self.loss_autoencoder =torch.nn.functional.mse_loss(decoded, x_all)
                    else:
                        self.loss_autoencoder +=torch.nn.functional.mse_loss(decoded, x_all)
                # this seems to be limiting how much BPTT happens.
                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, info)


            # this is actually giving you actions from logits
            action = select_action(self.args, action_out)
            # this is for the gating head penalty
            if not self.args.continuous:
                log_p_a = action_out
                p_a = [[z.exp() for z in x] for x in log_p_a]
                gating_probs = p_a[1][0].detach().numpy()

                # since we treat this as reward so probability of 0 being high is rewarded
                gating_head_rew = np.array([p[1] for p in gating_probs])
                if self.args.min_comm_loss and t != 0:
                    comm_prob = p_a[1][0]
                    comm_prob = comm_prob.T[1]
                    comm_losses = torch.zeros_like(comm_prob)
                    comm_losses[comm_prob < self.args.soft_budget] = (self.args.soft_budget - comm_prob[comm_prob < self.args.soft_budget]) / self.args.soft_budget
                    comm_losses[comm_prob >= self.args.soft_budget] = (comm_prob[comm_prob >= self.args.soft_budget] - self.args.soft_budget) / (1. - self.args.soft_budget)
                    comm_losses = torch.abs(comm_losses).mean()
                    if self.loss_min_comm == None:
                        self.loss_min_comm = comm_losses
                    else:
                        self.loss_min_comm += comm_losses
                if self.args.gating_head_cost_factor != 0:
                    if self.args.gating_punish:
                        # encourage communication to be at thresh %
                        # thresh = 0.125
                        thresh = self.args.soft_budget
                        Kp = 1.
                        # Kd = 3.2
                        Kd = 1.6
                        Ki = 0.026
                        Kpdi = 1.
                        # 0.05 is the minimum comm rate to ensure success
                        # gating_head_rew[gating_head_rew < 0.05] = 10
                        # error = (gating_head_rew - (0.5*(thresh_top+thresh_bot))) ** 2
                        error = np.zeros_like(gating_head_rew)
                        error[gating_head_rew < thresh] = (thresh - gating_head_rew[gating_head_rew < thresh]) / thresh
                        error[gating_head_rew >= thresh] = (thresh - gating_head_rew[gating_head_rew >= thresh]) / (1. - thresh)
                        if self.last_error is None:
                            self.last_error = error
                        derivative = error - self.last_error
                        if self.total_error is None:
                            self.total_error = np.zeros_like(error)
                        gating_head_rew = Kpdi * np.abs(Kp * error + Kd * derivative + Ki * self.total_error)
                        self.last_error = error
                        self.total_error += error
                        self.total_error = np.clip(self.total_error, -50, 50)
                        # gating_head_rew[gating_head_rew < 0.05] = (gating_head_rew[gating_head_rew < 0.05] - (0.5*(thresh_top+thresh_bot))) ** 2
                        # gating_head_rew[np.logical_and((gating_head_rew <= thresh_top), (gating_head_rew >= thresh_bot))] = 0
                        # gating_head_rew[gating_head_rew > 0.05] = (gating_head_rew[gating_head_rew > 0.05] - (0.5*(thresh_top+thresh_bot))) ** 2
                        # print("here punish", gating_head_rew, gating_probs)
                    else:
                        # encourage communication to be high
                        # gating_head_rew = (gating_head_rew - 1) ** 2
                        # gating_head_rew = (gating_head_rew - self.policy_net.budget) ** 2
                        # punish trying to communicate over budget scaled to [0,1]
                        # print(gating_head_rew, stat['comm_action'] / stat['num_steps'], info['comm_budget'])
                        if self.policy_net.budget != 1:
                            # gating_head_rew = (np.abs(info['comm_action'] - info['comm_budget'])).astype(np.float64)
                            # gating_head_rew = (np.abs(info['comm_action'] - info['comm_budget']) / (1 - self.policy_net.''budget'')).astype(np.float64)
                            # punish excessive and strengthen current communication
                            # gating_head_rew = (np.abs(gating_head_rew - info['comm_budget'])).astype(np.float64)
                            # only punish excessive communication
                            # mask_rew = info['comm_action'] != info['comm_budget']
                            # error = np.zeros_like(gating_head_rew)
                            # error[mask_rew] = np.abs(gating_head_rew[mask_rew] - info['comm_budget'][mask_rew]).astype(np.float64)
                            gating_head_rew = np.abs(gating_head_rew - info['comm_budget']).astype(np.float64)
                            # gating_head_rew = error
                        else:
                            # max communication when budget is full
                            # gating_head_rew = np.abs(info['comm_action'] - 1).astype(np.float64)
                            gating_head_rew = np.abs(gating_head_rew - 1).astype(np.float64)
                        # punish communication under budget scaled to [0,1]
                        # gating_head_rew += np.abs(info['comm_budget'] - self.policy_net.budget) / (self.policy_net.budget)
                        # print("here", gating_head_rew, gating_probs)
                    gating_head_rew *= -1 * np.abs(self.args.gating_head_cost_factor)
                    # try these methods:
                        # A) negative reward when not rate expected
                        # B) positive reward when <= comm rate expected, negative reward when > comm rate
                        # C) adaptive
                            # like A/B but comm rate is adaptive based on success rate
                    stat['gating_reward'] = stat.get('gating_reward', 0) + gating_head_rew
                    # print(gating_head_rew)

            # this converts stuff to numpy
            action, actual = translate_action(self.args, self.env, action)
            # decode intent + observation autoencoder
            if self.args.autoencoder and self.args.autoencoder_action:
                decoded = self.policy_net.decode()
                x_all = torch.zeros_like(decoded)
                x_all[0,:,:-self.args.nagents] = x[0].sum(dim=1).expand(self.args.nagents, -1)
                x_all[0,:,-self.args.nagents:] = torch.tensor(actual[0])
                if self.loss_autoencoder == None:
                    self.loss_autoencoder = torch.nn.functional.mse_loss(decoded, x_all)
                else:
                    self.loss_autoencoder += torch.nn.functional.mse_loss(decoded, x_all)
            comm_budget = info['comm_budget']
            next_state, reward, done, info = self.env.step(actual)

            stat['env_reward'] = stat.get('env_reward', 0) + reward[:self.args.nfriendly]
            if not self.args.continuous and self.args.gating_head_cost_factor != 0:
                if not self.args.variable_gate: # TODO: remove or True later
                    reward += gating_head_rew

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                info['step_t'] = t
                if self.args.comm_action_zero:
                    info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                stat['comm_budget'] = stat.get('comm_budget', 0) + comm_budget[:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask,
                               next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']
            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)

        return (episode, stat)

    def compute_grad(self, batch, other_stat=None):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)
        rewards = torch.Tensor(np.array(batch.reward)).to(self.device)
        episode_masks = torch.Tensor(np.array(batch.episode_mask)).to(self.device)
        episode_mini_masks = torch.Tensor(np.array(batch.episode_mini_mask)).to(self.device)
        actions = torch.Tensor(np.array(batch.action)).to(self.device)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0).to(self.device)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0).to(self.device) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(self.device)

        coop_returns = torch.Tensor(batch_size, n).to(self.device)
        ncoop_returns = torch.Tensor(batch_size, n).to(self.device)
        returns = torch.Tensor(batch_size, n).to(self.device)
        deltas = torch.Tensor(batch_size, n).to(self.device)
        advantages = torch.Tensor(batch_size, n).to(self.device)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])
        '''
	    coop_returns = rewards + self.args.gamma * prev_coop_return * episode_masks
        ncoop_returns = rewards + self.args.gamma * prev_ncoop_return * episode_masks * episode_mini_masks

        prev_coop_return = coop_returns.clone()
        prev_ncoop_return = ncoop_returns.clone()

        returns = (self.args.mean_ratio * coop_returns.mean()) \
                    + ((1 - self.args.mean_ratio) * ncoop_returns)

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]
        '''
        advantages = returns - values.data
        # print(advantages, returns, values.data,"\n")

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        # adding regularization term to minimize communication
        loss = action_loss + self.args.value_coeff * value_loss
        '''
        if self.args.min_comm_loss:
            comm_losses = other_stat['comm_action'] / float(other_stat['num_steps'])
            comm_losses = torch.Tensor(comm_losses).to(self.device).mean()
            comm_losses = torch.abs((self.args.soft_budget-comm_losses))
            if comm_losses < self.args.soft_budget:
                comm_losses /= self.args.soft_budget
            else:
                comm_losses /= (1. - self.args.soft_budget)
            comm_losses *= self.args.eta_comm_loss
            stat['regularization_loss'] = comm_losses.item()
            # print(stat['regularization_loss'], stat['action_loss'], self.args.value_coeff * stat['value_loss'])
            loss += comm_losses
        '''
        if self.args.max_info:
            loss += self.args.eta_info * 0   # TODO: add euclidean distance between memory cells

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        if self.args.min_comm_loss:
            self.loss_min_comm *= self.args.eta_comm_loss
            stat['regularization_loss'] = self.loss_min_comm.item()
            loss += self.loss_min_comm
        if self.args.autoencoder:
            stat['autoencoder_loss'] = self.loss_autoencoder.item()
            loss = 0.5 * loss + 0.5 * self.loss_autoencoder
        loss.backward()
        if self.args.autoencoder:
            self.loss_autoencoder = None
        if self.args.min_comm_loss:
            self.loss_min_comm = None

        # self.counter = 0
        # self.summer = 0
        # self.summer1 = 0
        return stat

    def run_batch(self, epoch):
        # self.reward_curriculum(epoch)
        if epoch >= 250 and self.args.use_tj_curric:
            self.begin_tj_curric = True
        self.epoch_num = epoch
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        self.stats['learning_rate'] = self.get_lr(self.optimizer)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        # run_st_time = time.time()
        batch, stat = self.run_batch(epoch)

        # print(f"time taken for data collection is {time.time() - run_st_time}")

        self.optimizer.zero_grad()

        # grad_st_time = time.time()
        s = self.compute_grad(batch, other_stat=stat)
        # print(f"time taken for grad computation {time.time() - grad_st_time}")

        merge_stat(s, stat)

        # Check if success has converged for curriculum learning
        self.success_curriculum(self.stats['success'], self.stats['num_episodes'])
        # check if time to introduce reward to decrease communication
        self.reward_curriculum(self.stats['success'], self.stats['num_episodes'])
        # increase spawn rate in traffic junction
        # self.tj_curriculum(self.stats['success'], self.stats['num_episodes'])
        # decrease hard limit of communication over time
        self.communication_curriculum(self.stats['success'], self.stats['num_episodes'])

        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()
        if self.args.scheduleLR:
            print("LR step")
            self.scheduler.step()
        return stat

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lrate

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def setup_var_reload(self):
        if self.args.variable_gate:
            self.args.comm_action_one = False
            self.args.variable_gate = False

    def load_scheduler(self, start_epoch):
        print("load_scheduler",start_epoch)
        self.scheduler1 = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1)
        self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer, 500*self.args.epoch_size, gamma=0.1)
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.scheduler1, self.scheduler2], milestones=[2500*self.args.epoch_size])
