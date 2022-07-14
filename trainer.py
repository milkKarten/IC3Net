from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import time

from timmac import TIMMAC
import os


Transition = namedtuple('Transition', ('state', 'action', 'action_out','comm_prob', 'comm_prob_logits', 'value', 'episode_mask',
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
                # self.success_thresh = .90
                self.success_thresh = .97
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
        self.best_model_reward = -np.inf
        self.kld_weight = 0.01
        self.recons_weight = 1.
        self.feature_loss = None

        if self.args.learn_past_comms:
            self.pretrained_policy_net = TIMMAC(self.args, self.args.num_inputs)
            load_path = os.path.join(self.args.load, self.args.env_name, self.args.past_comm_model_fp, "seed" + str(self.args.seed), "models")
            if 'best_model.pt' in os.listdir(load_path):
                model_path = os.path.join(load_path, "model.pt")
            else:
                assert False
            d = torch.load(model_path)
            self.pretrained_policy_net.load_state_dict(d['policy_net'], strict=False)
            self.pretrained_policy_net.eval()

    def get_episode(self, epoch, random=False):
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
        info_comm = dict()
        switch_t = -1

        # one is used because of the batch size.
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        if self.args.ic3net:
            stat['budget'] = self.policy_net.budget

        # episode_comm = torch.zeros(self.args.nagents)
        if self.args.timmac and not random:
            self.policy_net.reset()
        if random:
            inputs = []
        episode_comm = []
        n_alive_steps = 0
        n_alive_steps_per_agent = np.zeros(self.args.nagents)


        for t in range(self.args.max_steps):
            # print(t)
            misc = dict()
            if t == 0:
                info_comm["alive_mask"] = np.zeros(self.args.nagents)

            if t == 0 and self.args.hard_attn and self.args.commnet and not random:
                info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                # info_comm['comm_budget'] = np.zeros(self.args.nagents, dtype=int)
                info_comm['step_t'] = t  # episode step for resetting communication budget
                stat['comm_action'] = np.zeros(self.args.nagents, dtype=int)[:self.args.nfriendly]



            n_alive_steps_per_agent += info_comm["alive_mask"]
            n_alive_steps += sum(info_comm["alive_mask"])
            # recurrence over time
            if self.args.recurrent and not random:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                action_out, value, prev_hid, comm_prob, comm_prob_logits = self.policy_net(x, info_comm)
                # episode_comm += comm_action

                # this seems to be limiting how much BPTT happens.
                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if random:
                    inputs.append(x)
                action_out, value, comm_prob, comm_prob_logits = self.policy_net(x, info_comm)
            if self.args.autoencoder and not self.args.autoencoder_action and not random:
                decoded, log_var, mu, mmd_loss = self.policy_net.decode()
                if self.args.recurrent:
                    # x_all = x[0].reshape(-1).expand_as(decoded)
                    # x_all = x[0].sum(dim=1).expand(self.args.nagents, -1).reshape(decoded.shape)
                    x_all = x[0].expand(self.args.nagents,self.args.nagents, -1)
                else:
                    # x_all = x.reshape(-1).expand_as(decoded)
                    # x_all = x.sum(dim=1).expand(self.args.nagents, -1).reshape(decoded.shape)
                    x_all = x.expand(self.args.nagents,self.args.nagents, -1)

    
                if self.args.learn_past_comms:
                    self.pretrained_policy_net(x, info_comm)
                    if self.loss_autoencoder == None:
                        self.loss_autoencoder = torch.nn.functional.mse_loss(self.policy_net.encoded_info, self.pretrained_policy_net.encoded_info)
                    else:
                        self.loss_autoencoder += torch.nn.functional.mse_loss(self.policy_net.encoded_info, self.pretrained_policy_net.encoded_info)
                else:
                    #LOG COSH
                    # self.alpha = 1
                    # self.beta = 1
                    # recons_diff = decoded - x_all
                    # recons_loss = self.alpha * recons_diff + \
                    #                 torch.log(1. + torch.exp(- 2 * self.alpha * recons_diff)) - \
                    #                 torch.log(torch.tensor(2.0))

                    # recons_loss = recons_loss.mean()*(1. / self.alpha)

                    # if self.loss_autoencoder == None:
                    #     self.loss_autoencoder = recons_loss
                    # else:
                    #     self.loss_autoencoder += recons_loss

                    # kld_loss = self.kld_weight*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
                    # self.kld_loss = kld_loss
                    # self.loss_autoencoder += kld_loss
                    # self.mmd_loss = mmd_loss


                    # self.alpha = -0.5
        
                    # self.reg_weight = 100

                    # self.beta = 1

                    # self.alpha = 1
                    # self.beta = 1
                    # recons_diff = self.policy_net.get_features(decoded) - self.policy_net.get_features(x_all)
                    # recons_loss = self.alpha * recons_diff + \
                    #                 torch.log(1. + torch.exp(- 2 * self.alpha * recons_diff)) - \
                    #                 torch.log(torch.tensor(2.0))
                    # recons_loss = recons_loss.mean()*(1. / self.alpha)

                    if self.loss_autoencoder == None:
                        if self.args.load_feat_net:
                            self.loss_autoencoder = self.recons_weight*torch.nn.functional.mse_loss(self.policy_net.get_features(decoded),self.policy_net.get_features(x_all))
                            # self.loss_autoencoder = recons_loss
                        else:
                            self.loss_autoencoder = self.recons_weight*torch.nn.functional.mse_loss(decoded, x_all)
                        # self.loss_autoencoder = recons_loss
                    else:
                        if self.args.load_feat_net:
                            # self.loss_autoencoder += recons_loss
                            self.loss_autoencoder += self.recons_weight*torch.nn.functional.mse_loss(self.policy_net.get_features(decoded),self.policy_net.get_features(x_all))
                        else:
                            self.loss_autoencoder += self.recons_weight*torch.nn.functional.mse_loss(decoded, x_all)
                        # self.loss_autoencoder += recons_loss

                    if self.args.variational_enc and not self.args.load_feat_net:
                        kld_loss = self.kld_weight*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
                        self.kld_loss = kld_loss

                        # bias_corr = self.args.batch_size * (self.args.batch_size - 1)
                        self.loss_autoencoder += kld_loss + self.mmd_weight*mmd_loss
                        self.mmd_loss = mmd_loss
                        # self.mmd_loss = 0
                        # self.loss_autoencoder += kld_loss
                    if self.args.variational_enc and self.args.load_feat_net:
                        kld_loss = self.kld_weight*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
                        self.kld_loss = kld_loss
                        self.loss_autoencoder += kld_loss
                        self.mmd_loss = mmd_loss
                    
                    

            # mask action if not available
            #print(action_out, '\n', self.env.env.get_avail_actions())
            if hasattr(self.env.env, 'get_avail_actions'):
                avail_actions = np.array(self.env.env.get_avail_actions())
                action_mask = avail_actions==np.zeros_like(avail_actions)
                action_out[0, action_mask] = -1e10
                action_out = torch.nn.functional.log_softmax(action_out, dim=-1)



            # this is actually giving you actions from logits
            action = select_action(self.args, action_out)

            if self.args.learn_intent_gating and self.args.min_comm_loss:
                episode_comm.append(comm_prob.double().reshape(1,-1))

            # this is for the gating head penalty
            if not self.args.continuous and not self.args.comm_action_one:
                # log_p_a = action_out
                # p_a = [[z.exp() for z in x] for x in log_p_a]
                # gating_probs = p_a[1][0].detach().numpy()
                gating_probs = comm_prob.detach().numpy()

                # since we treat this as reward so probability of 0 being high is rewarded
                # gating_head_rew = np.array([p[1] for p in gating_probs])
                gating_head_rew = gating_probs
                if self.args.min_comm_loss:
                    # print("c prob", comm_prob)
                    episode_comm.append(comm_prob.double().reshape(1,-1))
                    # comm_prob = comm_prob.double()
                    # comm_losses = torch.zeros_like(comm_prob)
                    # ind_budget = np.ones(self.args.nagents) * self.args.max_steps * self.args.soft_budget
                    # ind_budget += np.ones(self.args.nagents) * self.policy_net.get_null_action()
                    # ind_budget = torch.tensor(ind_budget / self.args.max_steps)
                    # comm_losses[comm_prob < ind_budget] = (ind_budget[comm_prob < ind_budget] - comm_prob[comm_prob < ind_budget]) / ind_budget[comm_prob < ind_budget]
                    # comm_losses[comm_prob >= ind_budget] = (comm_prob[comm_prob >= ind_budget] - ind_budget[comm_prob >= ind_budget]) / (1. - ind_budget[comm_prob >= ind_budget])
                    # comm_losses = torch.abs(comm_losses).mean()
                    # if self.loss_min_comm == None:
                    #     self.loss_min_comm = comm_losses
                    # else:
                    #     self.loss_min_comm += comm_losses
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
            if self.args.autoencoder and self.args.autoencoder_action and not random:
                decoded, log_var, mu = self.policy_net.decode()
                x_all = torch.zeros_like(decoded)
                if self.args.recurrent:
                    # x_all[:,:-self.args.nagents] = x[0].sum(dim=1).expand(self.args.nagents, -1)
                    # x_all[0,:,:-self.args.nagents] = x[0].sum(dim=1).expand(1, self.args.nagents, -1)
                    # x_all[0,:,-self.args.nagents:] = torch.tensor(actual[0])
                     x_all = x[0].expand(self.args.nagents,self.args.nagents, -1)

                else:
                    # # decoded = decoded.reshape(decoded.shape[0],decoded.shape[1])
                    # x_all[:,:-self.args.nagents] = x.reshape(decoded.shape[0], decoded.shape[1]-self.args.nagents)
                    # # x_all[0,:,:-self.args.nagents] = x.sum(dim=1).expand(self.args.nagents, -1)
                    # x_all[:,-self.args.nagents:] = torch.tensor(actual[0])

                     x_all = x.expand(self.args.nagents,self.args.nagents, -1)

                gt_actions = torch.tensor(actual[0]).unsqueeze(1).expand(self.args.nagents,self.args.nagents,-1)
                x_all = torch.cat((x_all,gt_actions),dim=2)

                if self.loss_autoencoder == None:
                    self.loss_autoencoder = torch.nn.functional.mse_loss(decoded, x_all)
                else:
                    self.loss_autoencoder += torch.nn.functional.mse_loss(decoded, x_all)

                if self.args.variational_enc:
                    kld_loss = self.kld_weight*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 2))
                    self.kld_loss = kld_loss
                    self.loss_autoencoder += kld_loss

            # comm_budget = info_comm['comm_budget']
            next_state, reward, done, info = self.env.step(actual)

            stat['env_reward'] = stat.get('env_reward', 0) + reward[:self.args.nfriendly]
            if not self.args.continuous and self.args.gating_head_cost_factor != 0:
                if not self.args.variable_gate: # TODO: remove or True later
                    reward += gating_head_rew

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet and not random:
                # info_comm['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                info_comm['step_t'] = t
                if self.args.comm_action_zero:
                    info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                stat['comm_action'] = stat.get('comm_action', 0) + info_comm['comm_action'][:self.args.nfriendly]
                # stat['comm_budget'] = stat.get('comm_budget', 0) + comm_budget[:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info_comm['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
                info_comm["alive_mask"] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)
                info_comm["alive_mask"] = np.ones_like(reward)

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

            if self.args.learn_intent_gating:
                trans = Transition(state, action, action_out, comm_prob.detach().numpy(), comm_prob_logits.detach().numpy(), value, episode_mask, episode_mini_mask,
                                   next_state, reward, misc)
            else:
                trans = Transition(state, action, action_out, None, None, value, episode_mask, episode_mini_mask,
                                   next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if self.args.learn_feat_net:
            ep_states = [trans[0].double() for trans in episode]
            ep_acts = [torch.tensor(trans[1]).double() for trans in episode]

            # ep_states = [torch.zeros(trans[0].shape) for trans in episode]
            # ep_acts = [torch.zeros(1,5) for trans in episode]

            decoding_loss, temporal_dist_loss, forward_dyn_loss, inv_dyn_loss = self.policy_net.train_features(ep_states,ep_acts)
            
            if self.feature_loss is None:
                self.feature_loss = decoding_loss + 0.001*temporal_dist_loss + forward_dyn_loss + 0.01*inv_dyn_loss
            else:
                self.feature_loss += decoding_loss + 0.001*temporal_dist_loss + forward_dyn_loss + 0.01*inv_dyn_loss

            stat["feat_decoding_loss"] = decoding_loss.item()
            stat["temporal_dist_loss"] = temporal_dist_loss.item()
            stat["forward_dyn_loss"] = forward_dyn_loss.item()
            stat["inv_dyn_loss"] = inv_dyn_loss.item()
           

        if self.args.min_comm_loss:
            # episode_comm = torch.cat(episode_comm, 0).T
            # episode_comm = episode_comm.mean(1)
            #
            # for i in range(self.args.nagents):
            #     stat["agent" + str(i) + "_episode_comm"] = episode_comm[i].detach().item()
            #
            #
            # comm_losses = torch.zeros_like(episode_comm)
            # ind_budget = np.ones(self.args.nagents) * self.args.max_steps * self.args.soft_budget
            # # ind_budget += np.ones(self.args.nagents) * self.policy_net.get_null_action()
            # ind_budget = torch.tensor(ind_budget / self.args.max_steps)
            # comm_losses[episode_comm < ind_budget] = (ind_budget[episode_comm < ind_budget] - episode_comm[episode_comm < ind_budget]) / ind_budget[episode_comm < ind_budget]
            # comm_losses[episode_comm >= ind_budget] = (episode_comm[episode_comm >= ind_budget] - ind_budget[episode_comm >= ind_budget]) / (1. - ind_budget[episode_comm >= ind_budget])
            # comm_losses = stat['num_steps'] * torch.abs(comm_losses).mean()
            #

            episode_comm = torch.cat(episode_comm, 0).T
            n_alive_steps_per_agent = torch.tensor(n_alive_steps_per_agent).unsqueeze(1)

            mask = (n_alive_steps_per_agent != 0)

            episode_comm_temp = torch.divide(torch.sum(episode_comm,axis=1)[mask.squeeze()],n_alive_steps_per_agent[mask])
            episode_comm = torch.zeros(self.args.nagents)
            comm_losses = torch.zeros_like(episode_comm)
            # ind_budget = np.ones(self.args.nagents) * self.args.max_steps * self.args.soft_budget
            ind_budget = torch.zeros(self.args.nagents)

            mask_i = 0
            for i in range(self.args.nagents):
                if mask[i]:
                    episode_comm[i] = episode_comm_temp[mask_i]
                    ind_budget[i] = n_alive_steps * self.args.soft_budget
                    mask_i += 1

            if n_alive_steps != 0:
                ind_budget = torch.tensor(ind_budget / n_alive_steps)

            for i in range(self.args.nagents):
                stat["agent" + str(i) + "_episode_comm"] = episode_comm[i].detach().item()
                stat["agent" + str(i) + "_episode_budget"] = ind_budget[i].item()


            comm_losses[episode_comm < ind_budget] = (ind_budget[episode_comm < ind_budget] - episode_comm[episode_comm < ind_budget]) / ind_budget[episode_comm < ind_budget]
            comm_losses[episode_comm >= ind_budget] = (episode_comm[episode_comm >= ind_budget] - ind_budget[episode_comm >= ind_budget]) / (1. - ind_budget[episode_comm >= ind_budget])
            comm_losses = n_alive_steps * torch.abs(comm_losses).mean()


            if n_alive_steps == 0:
                comm_losses = 0

            if self.loss_min_comm == None:
                self.loss_min_comm = comm_losses
            else:
                self.loss_min_comm += comm_losses

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
        if random:
            return inputs
        return (episode, stat)

    def compute_grad(self, batch, other_stat=None):
        stat = dict()
        num_actions = self.args.num_actions
        # dim_actions = self.args.dim_actions
        dim_actions = 1

        n = self.args.nagents
        batch_size = len(batch.state)
        rewards = torch.Tensor(np.array(batch.reward)).to(self.device)
        episode_masks = torch.Tensor(np.array(batch.episode_mask)).to(self.device)
        episode_mini_masks = torch.Tensor(np.array(batch.episode_mini_mask)).to(self.device)
        actions = torch.Tensor(np.array(batch.action)).to(self.device)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        if self.args.learn_intent_gating:
            comm_actions = torch.Tensor(np.array(batch.comm_prob)).to(self.device).flatten().unsqueeze(1)
            comm_actions_logits = torch.Tensor(np.array(batch.comm_prob_logits)).to(self.device).flatten(end_dim=1)

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
                if self.args.learn_intent_gating:
                    comm_log_prob = multinomials_log_densities(comm_actions, comm_actions_logits)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)
                if self.args.learn_intent_gating:
                    comm_log_prob = multinomials_log_density(comm_actions, [comm_actions_logits])


        if self.args.advantages_per_action:
            if self.args.learn_intent_gating:
                action_loss = -advantages.view(-1).unsqueeze(-1) * (log_prob + comm_log_prob)
            else:
                action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            if self.args.learn_intent_gating:
                action_loss = -advantages.view(-1) * (log_prob.squeeze() + comm_log_prob.squeeze())
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
            if self.args.variational_enc and not self.args.learn_past_comms:
                stat['kld_loss'] = self.kld_loss.item()
                stat['mmd_loss'] = self.mmd_loss.item()
            loss = 0.5 * loss + 0.5 * self.loss_autoencoder

        if self.args.learn_intent_gating:
            for name, param in self.policy_net.named_parameters():
                if "gating_l" not in name:
                    param.requires_grad = False

        # loss = self.loss_min_comm
        if self.args.learn_feat_net:
            feat_net_params = ["feat_layer_1", "feat_decoder", "temporal_dist", "forward_dyn", "inv_dyn"]
            for name, param in self.policy_net.named_parameters():
                name_ = name.split(".")
                if name_[0] not in feat_net_params:
                    param.requires_grad = False

            loss = self.feature_loss.double()

        # loss = self.loss_autoencoder
        loss.backward()
        if self.args.autoencoder:
            self.loss_autoencoder = None
            if self.args.learn_feat_net:
                self.feature_loss = None
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

            # self.kld_weight += 1/self.args.batch_size

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
            # param_group['lr'] = self.args.lrate
            param_group['lr'] = self.args.lrate * 0.01

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
