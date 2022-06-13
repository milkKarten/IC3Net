import torch
import torch.nn.functional as F
from torch import nn
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
    def __init__(self, args, num_inputs, train_mode=True):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.continuous = args.continuous
        # If true, we add noise to the communication being output by each agent.
        self.add_comm_noise = args.add_comm_noise

        # TODO: remove this is just for debugging purposes just to verify that the communication is happening in a
        #  disrete manner
        self.unique_comms = []

        # defining mode which is useful in the case of prototype layers.
        self.train_mode = train_mode

        # Only really used when you're using prototypes
        self.exploration_noise = OUNoise(args.comm_dim)
        self.explore_choose_proto_noise = OUNoise(args.num_proto)

        # see if you're using discrete communication and using prototypes
        self.discrete_comm = args.discrete_comm
        # self.use_proto = args.use_proto

        # num_proto is not really relevant when use_proto is set to False
        self.num_proto = args.num_proto

        # this is discrete/proto communication which is not to be confused with discrete action. T
        # Although since the communication is being added to the encoded state directly, it makes things a bit tricky.
        if args.discrete_comm:
            self.proto_layer = ProtoNetwork(args.hid_size, args.comm_dim, args.discrete_comm, num_layers=2,
                                            hidden_dim=64, num_protos=args.num_proto, constrain_out=False)

        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            # self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
            #                             for o in args.naction_heads])
            self.action_head = nn.Linear(args.hid_size, args.num_actions[0])


        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            # this just prohibits self communication
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage

        # Note: num_inputs is 29 in the case Predator Prey.
        # TODO: Since currently you directly add the weighted hidden state to the encoded observation
        #  the output of the encoder is of the shape hidden. Basically we need to now make sure that in case of
        #  discrete also the dimension of the output of the state encoder is same as dimension of the output of the
        #  discrete communication.

        # self.encoder = nn.Linear(num_inputs, args.hid_size)

        # changed this for prototype based method. But should still work in the old case.
        self.encoder = nn.Linear(num_inputs, args.comm_dim)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        # TODO: currently the prototype is only being handled for the recurrent case. Do it more generally
        if args.recurrent:
            # not sure why is hidden dependent on batch size
            # also the initialised hiddens arent being assigned to anything
            self.init_hidden(args.batch_size)

            # Old code when the input size was equal to the hidden size.
            # self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)
            # comm, gating module
            # action module
            self.f_module = nn.LSTMCell(args.comm_dim, args.hid_size)


        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)

        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            # changed t
            # self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
            #                                 for _ in range(self.comm_passes)])

            self.C_modules = nn.ModuleList([nn.Linear(args.comm_dim, args.comm_dim)
                                            for _ in range(self.comm_passes)])

        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0

        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)
        self.gating_head = nn.Linear(self.hid_size, 2)

        self.value_head = nn.Linear(self.hid_size, 1)

        # communication limit, default always allows communication
        self.comm_budget = torch.tensor([self.args.max_steps+1] * self.nagents)
        self.budget = args.budget

        # autoencoder decoder
        if self.args.autoencoder_action:
            self.decoderNet = nn.Linear(args.hid_size, num_inputs+self.args.nagents)
        elif self.args.autoencoder:
            self.decoderNet = nn.Linear(args.hid_size, num_inputs)

        # remove null messages
        # with open('IC3Net/nulls/'+self.args.pretrain_exp_name+'/seed' + str(self.args.seed) + '/nulls.txt', 'r') as f:
        # with open('/Users/seth/Documents/research/neurips/nulls/tj_easy_proto_soft_minComm_autoencoder/seed' + str(self.args.seed) + '/nulls.txt', 'r') as f:
        # with open('/Users/seth/Documents/research/neurips/nulls/'+self.args.exp_name+'/seed' + str(self.args.seed) + '/nulls.txt', 'r') as f:
        if self.args.remove_null:
            null_path = os.path.join(self.args.null_dict_dir, self.args.exp_name, "seed" + str(self.args.seed), 'nulls.txt')
            with open(null_path) as f:
                protos = f.readlines()
                for i in range(len(protos)):
                    protos[i] = protos[i].replace("\n", "").split(',')
                self.null_dict = torch.tensor(np.array(protos).astype(np.float32))
                # for i in range(len(protos)):
                #     print(self.null_dict[i].shape)
        self.num_null = 0
        self.num_good_comms = 0
        self.num_cut_comms = 0
        self.num_comms = 0

        self.null_action = np.zeros(self.args.nagents)

        # Multi-head communication attention
        self.num_heads = args.num_heads
        # self.comm_mh_attn = nn.MultiheadAttention(args.hid_size, num_heads=self.num_heads)
        self.tokeys = nn.Linear(args.hid_size, args.hid_size*self.num_heads)
        self.toqueries = nn.Linear(args.hid_size, args.hid_size*self.num_heads)
        self.tovalues = nn.Linear(args.hid_size, args.hid_size*self.num_heads)
        self.unifyheads = nn.Linear(2 * args.hid_size, args.hid_size)

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x

            # In case of recurrent first take out the actual observation and then encode it.
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                # if you're using the extras would have both the hidden and the cell state.
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state

    def decode(self):
        y = self.h_state + self.comms_all
        y = self.decoderNet(y)
        return y

    def get_null_action(self):
        return self.null_action

    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)
        batch_size = x.size()[0]
        n = self.nagents
        # better comm generation
        x = x.view(batch_size * n, self.args.comm_dim)
        hidden_state, cell_state = self.f_module(x, (hidden_state, cell_state))

        if self.args.autoencoder:
            self.h_state = hidden_state.clone()

        # this should remain regardless of using prototypes or not.
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            # comm_action = torch.tensor(info['comm_action'])
            # comm_prob = comm_action
            # print(comm_action)
            comm_prob = None
            if self.args.comm_action_one:
                comm_action = torch.ones(self.nagents)
            elif self.args.comm_action_zero:
                comm_action = torch.zeros(self.nagents)
            else:
                h = hidden_state.view(batch_size, n, self.hid_size)
                # comm_prob = F.relu(self.gating_head(h))[0]
                # comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
                # print(comm_prob)
                comm_prob = F.log_softmax(self.gating_head(h), dim=-1)[0].exp()

                comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
                # print(comm_prob, torch.argmax(comm_prob, axis=-1))
                comm_prob = comm_prob[:, 1].reshape(self.nagents)
                # print(comm_prob)
                # comm_prob = torch.multinomial(comm_prob, 1).reshape(self.nagents)
                # print(comm_prob)
                comm_action = comm_prob
                # print(comm_action, comm_prob)
                # comm_prob = comm_prob[torch.arange(self.nagents), comm_action]
            # print("comm action", comm_action)
            #     comm_prob = torch.argmax(comm_action, axis=-1)
            #     # comm_prob = torch.max(comm_action, axis=-1).values
            #     # print(comm_action)
            #     comm_action = torch.argmax(comm_action, axis=-1).detach()
                # print(comm_action)
            # info['comm_action'] = comm_action.clone().detach().numpy()
            # print(comm_action)
            for c in range(self.args.nagents):
                if agent_mask[0,0,c] == 0: continue

            self.num_comms += num_agents_alive
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask = agent_mask * comm_action_mask.double()

        info['comm_action'] = comm_action.detach().numpy()

        agent_mask_transpose = agent_mask.transpose(1, 2)
        all_comms = []
        for i in range(self.comm_passes):
            if self.args.use_proto:
                raw_outputs = self.proto_layer(hidden_state)
                # raw_outputs is of shape (1, num_agents, num_protos). But we need to get rid of that first dimension.
                raw_outputs = torch.squeeze(raw_outputs, 0)
                if self.train_mode:
                    comm = self.proto_layer.step(raw_outputs, True, self.explore_choose_proto_noise, 'cpu')
                else:
                    comm = self.proto_layer.step(raw_outputs, False, None, 'cpu')
                    all_comms.append(comm.detach().clone())
                # Comm assumes shape (1, num_agents, num_protos), so just add that dimension back in.
                comm = torch.unsqueeze(comm, 0)

                if self.add_comm_noise:
                    # Currently, just hardcoded. We want enough noise to have an effect but not too much to prevent
                    # learning.
                    std = 0.2  # 0.4 for dim 16
                    # Generates samples from a zero-mean unit gaussian, which we rescale by the std parameter.
                    noise = torch.randn_like(comm) * std
                    comm += noise
                # check if comm contains null vector
                # print(comm.shape) # 1,5,64
                # sys.exit()
                # if self.args.null_regularization:

            elif self.args.discrete_comm:  #one-hot
                raw_outputs = self.proto_layer(hidden_state)
                raw_outputs = torch.squeeze(raw_outputs, 0)
                comm = self.proto_layer.onehot_step(raw_outputs, self.train_mode)
                all_comms.append(comm.detach().clone())
                comm = torch.unsqueeze(comm, 0)

            else:
                # print(f"inside else {hidden_state.size()}")
                comm = hidden_state
                # print("before", comm.shape, comm) # (5,32)
                all_comms.append(torch.squeeze(comm, 0).detach().clone())
                assert self.args.comm_dim == self.args.hid_size , "If not using protos comm dim should be same as hid"

            if self.args.remove_null:
                null_mask = torch.ones_like(comm)
                for j in range(self.args.nagents):
                    if agent_mask[0,0,j] == 0:
                        continue
                    found_null = False
                    for null_i in range(len(self.null_dict)):
                        threshold = 0.1
                        if not self.args.discrete_comm:
                            threshold = 1.
                        if torch.nn.functional.mse_loss(self.null_dict[null_i], comm[0,j]) < threshold:
                            null_mask[0,j] *= 0
                            found_null = True
                            break
                    if not found_null:
                        # track non null communicated
                        if info['comm_action'][j] == 1:
                            self.num_good_comms += 1
                    # else:
                    #     if info['comm_action'][j] == 0:
                    #         self.num_null += 1
                self.null_action = np.zeros(self.args.nagents)
                if 'null' in self.args.exp_name or True:
                    for j in range(self.args.nagents):
                        if null_mask[0,j].sum() == 0:
                            if info['comm_action'][j] == 1:    # we cut an additional communication
                                self.null_action[j] = 1 # get one comm back for later
                                self.num_null += 1
                                self.num_cut_comms += 1
                    comm = comm * null_mask
            # comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm  = comm.view(batch_size, n, self.args.comm_dim) if self.args.recurrent else comm
            # Get the next communication vector based on next hidden state
            # comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

            # changed for accommodating prototype based approach as well.
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.args.comm_dim)

            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)

            # mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            mask = mask.expand_as(comm)
            comm = comm * mask

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            if self.args.mha_comm:
                # Multi-head attention for incoming comms
                comm = comm.reshape(n, n, self.args.comm_dim)
                sh = (n * self.num_heads, n, self.args.comm_dim)
                Q = self.toqueries(comm).view(sh)
                K = self.tokeys(comm).view(sh)
                V = self.tovalues(comm).view(sh)
                self.norm_factor = 1 / np.sqrt(self.args.comm_dim)
                Q = Q * self.norm_factor
                K = K * self.norm_factor
                dot = torch.bmm(Q, K.transpose(1, 2))
                assert dot.size() == (n * self.num_heads, n, n)
                # mask again before softmax
                dot_mask = (mask*agent_mask*agent_mask_transpose).reshape(n,n,-1)[:,:,:n]
                # print(int(dot.shape[0]/dot_mask.shape[0]))
                dot_mask = dot_mask.repeat_interleave(repeats=int(dot.shape[0]/dot_mask.shape[0]), dim=0)
                # print(dot_mask.shape, dot.shape)

                dot = dot * dot_mask
                dot = dot.masked_fill(dot_mask == 0, -1e9)

                attn = F.softmax(dot, dim=-1)
                out = torch.bmm(attn, V).view(n, self.num_heads, n, self.args.comm_dim)
                out = out.transpose(1, 2).contiguous().view(n, n, self.num_heads * self.args.comm_dim)
                comm = out.sum(dim=0)
                comm = self.unifyheads(torch.cat((comm, hidden_state), -1)).reshape(batch_size, n, self.args.comm_dim)
                comm = self.tanh(comm)
                # print(comm.shape)
                assert comm.size() == (batch_size, n, self.args.comm_dim)
                c = comm

            else:
                # print("comm mode ", self.args.comm_mode)
                if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                    and num_agents_alive > 1:
                    comm = comm / (num_agents_alive - 1)
                # Combine all of C_j for an ith agent which essentially are h_j
                comm_sum = comm.sum(dim=1)

                c = self.C_modules[i](comm_sum)
            if self.args.autoencoder:
                self.comms_all = c.clone()  # encoded received communciations for autoencoder

            if self.args.recurrent:
                if self.args.mha_comm:
                    inp = c
                else:
                    # skip connection - combine comm. matrix and encoded input for all agents
                    inp = hidden_state + c

                # inp = inp.view(batch_size * n, self.hid_size)

                inp = inp.view(batch_size * n, self.args.comm_dim)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            # action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
            if self.args.env_name == 'starcraft':
                action = self.action_head(h)
            else:
                action = F.log_softmax(self.action_head(h), dim=-1)
            # print(f"uses discrete actions {action}")
        if self.args.recurrent:
            if info.get('record_comms') is not None:
                # Go through the all comms passes and only pick out comms for the agent you want.
                # filtered_comms = np.array([c[info.get('record_comms')] for c in all_comms])
                filtered_comms = np.array([c.numpy() for c in all_comms])
                if self.args.env_name == 'predator_prey':
                    assert len(filtered_comms) == 1, "Only support one agent at a time"
                # print("communication comm.py", c.shape, len(filtered_comms[0]))
                # print(info['comm_action'])
                return action, value_head, (hidden_state.clone(), cell_state.clone()), filtered_comms
            return action, value_head, (hidden_state.clone(), cell_state.clone()), comm_prob
        else:
            if info.get('record_comms') is not None:
                filtered_comms = [c[info.get('record_comms')] for c in all_comms]
                assert len(filtered_comms) == 1, "Only support one agent at a time"
                return action, value_head, filtered_comms[0]
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
