from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
from ic3net_envs import predator_prey_env
import cv2

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Evaluator:
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = args.display
        self.last_step = False
        self.episonde_n = 0
        self.car2color = [(0,0,255),(255,0,0),(255, 165, 0),(230,230,250),(0,255,0)]
        self.car_imgs = [cv2.imread("car_imgs/car_" + str(i) + ".png") for i in range(5)]
        self.seed2act = {}
        self.seed2alive = {}

    def save_action(self,a,seed):
        alive_vec = [self.env.get_alive_wrapper(i) for i in range(self.args.nagents)]
        if seed in self.seed2act:
            self.seed2act[seed].append(a)
            self.seed2alive[seed].append(alive_vec)
        else:
            self.seed2act[seed] = [a]
            self.seed2alive[seed] = [alive_vec]

    def run_episode(self, epoch=1):
        self.episonde_n += 1

        episode_frames = []


        all_comms = []
        episode = []
        reset_args = getargspec(self.env.reset).args

        if "special" in epoch:
            epoch_ = epoch.split("_")
            self.env.env.seed(int(epoch_[1]))
        else:
            self.env.env.seed(epoch)
        # print (epoch)
        if 'epoch' in reset_args:
            # print ("reset args")
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display  # and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        stat["autoencoder_loss"] = 0
        stat["n_loss_checks"] = 0
        stat["loc_pred_acc"] = 0
        stat["loc_pred_checks"] = 0
        info_comm = dict()
        info= dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        comms_to_prey_loc = {} # record action 0
        comms_to_prey_act = {} # record action 1
        comms_to_loc_full = {} # record all
        comm_action_episode = np.zeros(self.args.max_steps)
        sas_trips = []
        for t in range(self.args.max_steps):
            misc = dict()
            info['step_t'] = t
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)
            # Hardcoded to record communication for agent 1 (prey)
            info['record_comms'] = 1
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]

                # if "special" in epoch:
                #     #disable communication after the first time step
                #     x[1] = self.policy_net.init_hidden(batch_size=state.shape[0])
                #     info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)

                if "special" in epoch and t > 20:
                    print ("NO MORE COMMUNICATION")
                    action_out, value, prev_hid, proto_comms = self.policy_net(x, info_comm,no_comm=True)
                else:
                    # print ("agent mask: ")
                    # if 'alive_mask' in info:
                    #     info_comm["alive_mask"] = info["alive_mask"]
                    #     print (info["alive_mask"])
                    #     print ([self.env.get_alive_wrapper(i) for i in range(self.args.nagents)])
                    #     print ("\n")
                    # else:
                    #     print ("no alive mask")

                    print ("COMMUNICATED NEXT ACTION")
                    print (info_comm)
                    action_out, value, prev_hid, proto_comms = self.policy_net(x, info_comm)
                # print (action_out)
                # print (x[1])
                # print ("\n")
                # if isinstance(self.env.env.env, predator_prey_env.PredatorPreyEnv):
                if self.args.env_name == 'predator_prey':
                    # tuple_comms = tuple(proto_comms.detach().numpy())
                    for i in range(0, len(self.env.env.predator_loc)):
                        p = self.env.env.predator_loc[i]
                        proto = proto_comms[0][i]
                        if info_comm['comm_action'][i] == 0 or self.policy_net.get_null_action()[i] == 1:
                            continue
                        tuple_comms = tuple(proto)
                        if comms_to_loc_full.get(tuple_comms) is None:
                            comms_to_loc_full[tuple_comms] = []
                        comms_to_loc_full[tuple_comms].append(tuple(p))
                elif self.args.env_name == 'traffic_junction':
                    # print("car loc", self.env.env.car_loc)
                    # print("paths", self.env.env.car_loc)
                    for i in range(0, len(self.env.env.car_loc)):
                        p = self.env.env.car_loc[i]
                        # print(p)
                        continue
                        proto = proto_comms[0][i]
                        action_i = self.env.env.car_last_act[i]
                        if self.env.env.car_route_loc[i] != -1:
                            if p[0] == 0 and p[1] == 0 or info_comm['comm_action'][i] == 0 or self.policy_net.get_null_action()[i] == 1:
                                continue
                            # print("path", p, proto.shape)
                            # print(t, "proto", proto, proto.shape)
                            # print(info_comm['comm_action'][i])
                            tuple_comms = tuple(proto)
                            # print("tuple comms", proto.shape)
                            if comms_to_loc_full.get(tuple_comms) is None:
                                comms_to_loc_full[tuple_comms] = []
                            comms_to_loc_full[tuple_comms].append(tuple(p))
                            # print(action_i)
                            if action_i == 0:
                                if comms_to_prey_loc.get(tuple_comms) is None:
                                    comms_to_prey_loc[tuple_comms] = []
                                # print("path", self.env.env.chosen_path[0])
                                comms_to_prey_loc[tuple_comms].append(tuple(p))
                            else:
                                if comms_to_prey_act.get(tuple_comms) is None:
                                    comms_to_prey_act[tuple_comms] = []
                                comms_to_prey_act[tuple_comms].append(tuple(p))


                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value, proto_comms = self.policy_net(x, info_comm)
                # if isinstance(self.env.env.env, predator_prey_env.PredatorPreyEnv):
                if self.args.env_name == 'predator_prey':
                    tuple_comms = tuple(proto_comms.detach().numpy())
                    if comms_to_prey_loc.get(tuple_comms) is None:
                        comms_to_prey_loc[tuple_comms] = []
                    comms_to_prey_loc[tuple_comms].append(tuple(self.env.env.env.prey_loc[0]))
                elif self.args.env_name == 'traffic_junction':
                    # print("car loc", self.env.env.car_loc)
                    # print("paths", self.env.env.car_loc)
                    for i in range(0, len(self.env.env.car_loc)):
                        p = self.env.env.car_loc[i]
                        # print(p)
                        proto = proto_comms[0][i]
                        action_i = self.env.env.car_last_act[i]
                        if self.env.env.car_route_loc[i] != -1:
                            # print("path", p, proto.shape)
                            tuple_comms = tuple(proto)
                            # print("tuple comms", proto.shape)
                            if comms_to_loc_full.get(tuple_comms) is None:
                                comms_to_loc_full[tuple_comms] = []
                            comms_to_loc_full[tuple_comms].append(tuple(p))

                            # print(action_i)
                            if action_i == 0:
                                if comms_to_prey_loc.get(tuple_comms) is None:
                                    comms_to_prey_loc[tuple_comms] = []
                                # print("path", self.env.env.chosen_path[0])
                                comms_to_prey_loc[tuple_comms].append(tuple(p))
                            else:
                                if comms_to_prey_act.get(tuple_comms) is None:
                                    comms_to_prey_act[tuple_comms] = []
                                comms_to_prey_act[tuple_comms].append(tuple(p))


            if hasattr(self.env.env, 'get_avail_actions'):
                avail_actions = np.array(self.env.env.get_avail_actions())
                action_mask = avail_actions==np.zeros_like(avail_actions)
                action_out[0, action_mask] = -1e10
                action_out = torch.nn.functional.log_softmax(action_out, dim=-1)


            prev_car_loc = self.env.env.car_loc
            prev_state = state.clone()


            action = select_action(self.args, action_out, eval_mode=True)
            action, actual = translate_action(self.args, self.env, action)
            # self.save_action(action,epoch)


            if self.args.env_name == 'traffic_junction':
                if self.args.train_fdm:
                    sas_trips.append([x[0].squeeze().tolist(),actual[0].tolist()])
                    if t != 0:
                        sas_trips[t-1].append(x[0].squeeze().tolist())
            if self.args.env_name == 'traffic_junction':
                if self.args.autoencoder and self.args.autoencoder_action:
                    decoded = self.policy_net.decode()


                    x[0] = x[0].squeeze()
                    # is_alive_vector = torch.tensor([1-self.env.get_alive_wrapper(i) for i in range(self.args.nagents)]).unsqueeze(0).T
                    #
                    # x[0] = torch.cat((x[0],is_alive_vector),dim=1)
                    # #swap last two columns
                    # x[0][:,[-1,-2]] = x[0][:,[-2,-1]]

                    # x_all = x[0].expand(self.args.nagents,self.args.nagents, -1)
                    # gt_actions = torch.tensor(actual[0]).unsqueeze(1).expand(self.args.nagents,self.args.nagents,-1)
                    # x_all = torch.cat((x_all,gt_actions),dim=2)
                    # # x_all = torch.zeros_like(decoded)
                    # # x_all[0,:,:-self.args.nagents] = x[0].sum(dim=1).expand(self.args.nagents, -1)
                    # # x_all[0,:,-self.args.nagents:] = torch.tensor(actual[0])
                    # #
                    #
                    # loss_autoencoder = torch.nn.functional.mse_loss(decoded, x_all).detach().numpy()
                    # stat["autoencoder_loss"] += loss_autoencoder
                    # stat["n_loss_checks"] += 1
                    loss_autoencoder = 0

                    decoded = decoded.detach()
                    decoded = decoded.squeeze()
                    player_decoded = decoded[0]

                    if self.args.comm_intent_1 or self.args.comm_intent_2:
                        player_decoded_locs = player_decoded[0:,2:-2-(self.args.intent_horizon-1)]
                        player_decoded_intent = player_decoded[0:,-2-(self.args.intent_horizon-1):-2]
                    else:
                        player_decoded_locs = player_decoded[0:,2:-2]

                    player_decoded_ris = player_decoded[0:,1].detach().numpy()
                    player_decoded_prev_actions = player_decoded[0:,0].detach().numpy()
                    player_decoded_actions = player_decoded[1:,-1].detach().numpy()

                    player_decoded_actions = np.where(player_decoded_actions > 1, 1, 0)

                    # player_decoded_locs = player_decoded_locs.tolist()
                    # player_decoded_locs_ = player_decoded_locs.clone()
                    player_decoded_locs = torch.nn.functional.softmax(player_decoded_locs/0.0001,dim=1).tolist()

                    # player_decoded_locs_ = torch.nn.functional.softmax(player_decoded_locs_,dim=1).tolist()
                    # print (player_decoded_locs_)
                    # print (sum(player_decoded_locs_[0]))
                    # assert False
                    # print (player_decoded_locs_)
                    # stat["mean_highest_oc_prob"] = np.mean(player_decoded_locs_,axis=1)

                    display_decoded_state = True

                    if display_decoded_state:

                        frame = np.ones((620,1650,3),np.uint8)*255

                        #draw roads
                        w, h = self.env.env.dims
                        grid = self.env.get_grid_wrapper()
                        for grid_i in range(h):
                            for grid_j in range(w):
                                if grid[grid_i,grid_j] == 1:
                                    x = grid_i*50+40
                                    y = grid_j*50+40
                                    cv2.rectangle(frame,(x,y),(x+70,y+50),(150,150,150),-1)
                                    cv2.rectangle(frame,(x+670,y),(x+70+670,y+50),(150,150,150),-1)


                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, 'GT State', (170,30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, 'Decoded State', (800,30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, 'AE Loss: ' + str(np.round(loss_autoencoder,3)), (490,250), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


                        cv2.rectangle(frame,(40,30+10),(410,390),(0,0,0),3)
                        cv2.rectangle(frame,(40+670,30+10),(410+670,390),(0,0,0),3)
                        h, w,_ =  self.car_imgs[i].shape

                    for i in range(0, len(self.env.env.car_loc)):
                        player_decoded_locs[i] = np.round(player_decoded_locs[i],0)


                        if self.args.comm_intent_1 or self.args.comm_intent_2:
                            player_decoded_intent_ = player_decoded_intent[i].detach().float().numpy()
                            player_decoded_intent_ = ["go" if np.round(int_,0) == 0 else "brake" for int_ in player_decoded_intent_]
                            # print (player_decoded_locs[i])

                        pred_loc = self.env.obs2pos_wrapper(tuple(player_decoded_locs[i]))
                        if pred_loc is None:
                            # print (player_decoded_locs[i])
                            pred_loc = [0,0]

                        state_ = state.squeeze().numpy()

                        # p = self.env.env.car_loc[i]
                        p = self.env.obs2pos_wrapper(tuple(state_[i][2:-1]))

                        if p is None:
                            p = [0,0]

                        if self.env.get_alive_wrapper(i):
                            if np.array_equal(p,pred_loc):
                                stat["loc_pred_acc"] += 1
                            stat["loc_pred_checks"] +=1

                        if display_decoded_state:
                            x = p[0]*50+50
                            y = p[1]*50+50
                            pred_x = pred_loc[0]*50+50+670
                            pred_y = pred_loc[1]*50+50

                            #draw agent locations if alive
                            if self.env.get_alive_wrapper(i):
                                cv2.rectangle(frame,(x,y),(x+50,y+20),self.car2color[i],-1)
                            # if pred_loc[0] + pred_loc[1] != 0:
                                cv2.rectangle(frame,(pred_x,pred_y),(pred_x+50,pred_y+20),self.car2color[i],-1)

                            #draw key to show agent stats
                            key_end= 30+50,460+20  + i*30
                            key_start = 30,460 + i*30
                            cv2.rectangle(frame,key_start,key_end,self.car2color[i],-1)

                            # print (next_state_[i])
                            prev_act = "go" if state_[i][0] == 0 else "brake"

                            curr_act = "go" if actual[0][i] == 0 else "brake"

                            cv2.putText(frame, 'Route ID: ' + str(state_[i][1]) + ", Prev Act: " + str(prev_act) + ", Curr Act: " + str(curr_act),(key_end[0]+10,key_end[1]-3), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)



                            dec_key_end = (30+670+50,460+20  + i*30)
                            dec_key_start = (30+670,460 + i*30)

                            dec_prev_act = "go" if player_decoded_prev_actions[i] < 1 else "brake"
                            dec_curr_act = "go" if player_decoded_prev_actions[i] < 1 else "brake"
                            cv2.rectangle(frame,dec_key_start,dec_key_end,self.car2color[i],-1)

                            if self.args.comm_intent_1 or self.args.comm_intent_2:
                                cv2.putText(frame, 'Route ID: ' + str(np.round(player_decoded_ris[i],3)) + ", Prev Act: " + str(dec_prev_act) + ", Curr Act: " + str(dec_curr_act)+ ", Intent: " + str(player_decoded_intent_),(dec_key_end[0]+10,dec_key_end[1]-3), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, 'Route ID: ' + str(np.round(player_decoded_ris[i],3)) + ", Prev Act: " + str(dec_prev_act) + ", Curr Act: " + str(dec_curr_act),(dec_key_end[0]+10,dec_key_end[1]-3), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                    if display_decoded_state:
                        cv2.imshow("frame",frame)
                        cv2.waitKey(0)
                        # print ("\n")
                        # cv2.imwrite("decoding_tests/episode" + str(self.episonde_n) + "/frame"+str(t)+".png",frame)

                        # assert False


            print ("going to take action: ")
            print (actual)
            next_state, reward, done, info = self.env.step(actual)
            print ("\n")
            done = done or self.env.env.has_failed
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                # info_comm['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                # print(info_comm['comm_action'][0])
                comm_action_episode[t] += info_comm['comm_action'][0]
                # print("before ", stat.get('comm_action', 0), info_comm['comm_action'][:self.args.nfriendly])
                stat['comm_action'] = stat.get('comm_action', 0) + info_comm['comm_action'][:self.args.nfriendly]
                all_comms.append(info_comm['comm_action'][:self.args.nfriendly])
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info_comm['comm_action'][self.args.nfriendly:]


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

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break

        if self.args.train_fdm:

            del sas_trips[-1]

            fdm_states = torch.tensor([sas[0] for sas in sas_trips])
            fdm_acts = torch.tensor([sas[1] for sas in sas_trips]).unsqueeze(2)
            fdm_next_states = torch.tensor([sas[2] for sas in sas_trips])

            fdm_sa_pair = torch.cat((fdm_states, fdm_acts),2)

            fdm_sa_pair = torch.flatten(fdm_sa_pair,end_dim=1)
            fdm_next_states = torch.flatten(fdm_next_states,end_dim=1)

            pred_next_states = self.policy_net.fdm(fdm_sa_pair)

            fdm_n_correct = 0
            total_fdm_trials = 0
            for pred_next_state, fdm_next_state in zip(pred_next_states, fdm_next_states):
                if sum(fdm_next_state) == 0:
                    continue
                pred_next_state = torch.nn.functional.softmax(pred_next_state[2:-1]/0.0001).tolist()
                pred_next_state = np.round(pred_next_state,0)

                if np.array_equal(pred_next_state, fdm_next_state[2:-1].tolist()):
                    fdm_n_correct += 1

                total_fdm_trials+=1
            if total_fdm_trials != 0:
                print (fdm_n_correct/total_fdm_trials)
                # stat['fdm_acc'] = (fdm_n_correct/total_fdm_trials)

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
        return episode, stat, all_comms, comms_to_prey_loc, comms_to_prey_act, comms_to_loc_full, comm_action_episode
