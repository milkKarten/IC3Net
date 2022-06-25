import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch import optim
import time
from models import MLP
from action_utils import select_action, translate_action
from networks import ProtoNetwork, ProtoLayer
from network_utils import gumbel_softmax
from noise import OUNoise
import numpy as np
import os




class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self):
        super(CommNetMLP, self).__init__()
        self.nagents = 5
        self.budget_measured_t = 0
        self.remaining_budget = torch.zeros(1, self.nagents,1)
        self.budget_enforced_t = 20
        self.budget = 0.9
        self.gating_l1 = nn.Linear(1, 2)


    def gating_func(self,current_comm,print_info=False):

        if self.budget_measured_t == 0:
            self.remaining_budget = torch.zeros(1,self.nagents,1)

        # gating_in = torch.cat((current_comm, self.last_recieved_comms, self.last_recieved_ts,self.remaining_budget),dim=2)
        # in_ = torch.tensor(self.budget_measured_t).unsqueeze(0).double()

        gating_out = self.gating_l1(self.remaining_budget.clone())


        comm_prob = F.log_softmax(gating_out, dim=-1)[0]


        if print_info:
            print ("comm_prob: ", comm_prob)

        # comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
        comm_prob = nn.functional.gumbel_softmax(comm_prob,hard=True,dim=-1)
        comm_prob = comm_prob[:, 1].reshape(self.nagents)

        if print_info:
            print ("sampled comm_prob", comm_prob)

        self.budget_measured_t += 1
        self.budget_measured_t = self.budget_measured_t % self.budget_enforced_t

        return comm_prob

    def forward(self,print_info=False):
        comm_prob = self.gating_func(None,print_info)

        comm_indices = torch.nonzero(comm_prob).squeeze()
        if comm_indices.dim() != 0:
            for comm_indice in comm_indices:
                self.remaining_budget[0][comm_indice] += 1/(self.budget*self.budget_enforced_t)
        return comm_prob

def train():
    nagents = 5
    budget = 0.3
    max_steps = 20

    policy_net = CommNetMLP()
    optimizer = optim.RMSprop(policy_net.parameters(),lr = 0.03, alpha=0.97, eps=1e-6)
    agent_coms = []
    for episode in range(200):
        episode_comm = []
        for step in range(20):
            comm_prob = policy_net()
            episode_comm.append(comm_prob.double().reshape(1,-1))

        episode_comm = torch.cat(episode_comm, 0).T
        episode_comm = episode_comm.mean(1)

        ep_agent_coms = np.zeros(nagents)
        for i in range(nagents):
            ep_agent_coms[i] = episode_comm[i].detach().item()

        agent_coms.append(ep_agent_coms)

        comm_losses = torch.zeros_like(episode_comm)
        ind_budget = np.ones(nagents) * max_steps * budget
        ind_budget += np.ones(nagents) * 0
        ind_budget = torch.tensor(ind_budget / max_steps)

        comm_losses[episode_comm < ind_budget] = (ind_budget[episode_comm < ind_budget] - episode_comm[episode_comm < ind_budget]) / ind_budget[episode_comm < ind_budget]
        comm_losses[episode_comm >= ind_budget] = (episode_comm[episode_comm >= ind_budget] - ind_budget[episode_comm >= ind_budget]) / (1. - ind_budget[episode_comm >= ind_budget])
        # comm_losses = stat['num_steps'] * torch.abs(comm_losses).mean()
        comm_losses = torch.abs(comm_losses).mean()

        optimizer.zero_grad()
        comm_losses.backward()
        optimizer.step()

    print (ep_agent_coms)
    print (comm_losses)

    print ("======= TEST EPISOD =======")
    for step in range(20):
        policy_net.budget_measured_t = 0
        comm_prob = policy_net(True)
        print ("\n")


train()
