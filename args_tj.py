import argparse

def get_args():

     # TODO: Run proto version
     # explicity add to reward function to brake before intersection?
     # does it make sense to have two way communication before action
     # interleave a small amount of supervised data with some self-play
     # send email to dana to setup box
     # try keeping spawning rate constant
     env = "traffic_junction"
     # seeds = [1, 2]
     seeds = [777]
     # seeds = [20]
     # your models, graphs and tensorboard logs would be save in trained_models/{exp_name}
     methods = ["fixed_proto_medium_noise"]
     # run baseline with no reward on the gating function
     # G - IC3net with learned gating function
     # exp_name = "tj_g0.01_test"
     # for reward_curr_start, reward_curr_end in zip([1500, 1250, 1800],[1900, 2000, 2000]):
     # for num_proto in [112]:
     if True:
          for method in methods:
               exp_name = "tj_human_" + method
               vision = 0
               # discrete comm is true if you want to use learnable prototype based communication.
               discrete_comm = False
               if "proto" in method:
                    discrete_comm = True
               num_epochs = 3000
               hid_size = 128
               save_every = 100
               # g=1. If this is set to true agents will communicate at every step.
               comm_action_one = False
               comm_action_zero = False
               # weight of the gating penalty. 0 means no penalty.
               # gating_head_cost_factor = rew
               gating_head_cost_factor = 0.1
               if "baseline" in method:
                    gating_head_cost_factor = 0
               if "fixed" in method:
                    if not "var" in method:
                         gating_head_cost_factor = 0
                    comm_action_one = True
               # specify the number of prototypes you wish to use.
               num_proto = 112  # try to increase prototypes
               # dimension of the communication vector.
               comm_dim = 64 # for explainability
               # if "bigcomm" in method:
               #     comm_dim = 32
               if not discrete_comm:
                    comm_dim = hid_size
               # use reward curriculum
               reward_curriculum = False
               if "rew_cur" in method:
                    reward_curriculum = True
               variable_gate = False
               if "var" in method:
                    variable_gate = True
               nprocesses = 1
               lr = 0.001
               if "medium" in method:
                    nagents = 10
                    max_steps = 40
                    dim = 14
                    add_rate_min = 0.05
                    add_rate_max = 0.2
                    difficulty = 'medium'
               elif "hard" in method:
                    nagents = 20
                    max_steps = 80
                    dim = 18
                    add_rate_min = 0.02
                    add_rate_max = 0.05
                    difficulty = 'hard'
               elif "longer_easy" in method:
                    nagents = 10
                    max_steps = 30
                    dim = 14
                    add_rate_min = 0.1
                    add_rate_max = 0.3
                    difficulty = 'longer_easy'
               else:
                    # easy
                    nagents = 5
                    max_steps = 20
                    dim = 6
                    add_rate_min = 0.1
                    add_rate_max = 0.3
                    difficulty = 'easy'

     # parser = argparse.ArgumentParser(description='PyTorch RL trainer')
     parser = argparse.ArgumentParser(description='PyTorch RL trainer')
     # training
     # note: number of steps per epoch = epoch_size X batch_size x nprocesses
     parser.add_argument('--num_epochs', default=num_epochs, type=int,
                         help='number of training epochs')
     parser.add_argument('--epoch_size', type=int, default=10,
                         help='number of update iterations in an epoch')
     parser.add_argument('--batch_size', type=int, default=500,
                         help='number of steps before each update (per thread)')
     parser.add_argument('--nprocesses', type=int, default=nprocesses,
                         help='How many processes to run')
     # model
     parser.add_argument('--hid_size', default=hid_size, type=int,
                         help='hidden layer size')
     parser.add_argument('--recurrent', action='store_true', default=True,
                         help='make the model recurrent in time')
     # optimization
     parser.add_argument('--gamma', type=float, default=1.0,
                         help='discount factor')
     parser.add_argument('--tau', type=float, default=1.0,
                         help='gae (remove?)')
     parser.add_argument('--seed', type=int, default=seeds[0],
                         help='random seed. Pass -1 for random seed') # TODO: works in thread?
     parser.add_argument('--normalize_rewards', action='store_true', default=False,
                         help='normalize rewards in each batch')
     parser.add_argument('--lrate', type=float, default=lr,
                         help='learning rate')
     parser.add_argument('--entr', type=float, default=0,
                         help='entropy regularization coeff')
     parser.add_argument('--value_coeff', type=float, default=0.01,
                         help='coeff for value loss term')
     # environment
     parser.add_argument('--env_name', default=env,
                         help='name of the environment to run')
     parser.add_argument('--max_steps', default=max_steps, type=int,
                         help='force to end the game after this many steps')
     parser.add_argument('--nactions', default='1', type=str,
                         help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
     parser.add_argument('--action_scale', default=1.0, type=float,
                         help='scale action output from model')
     # other
     parser.add_argument('--plot', action='store_true', default=False,
                         help='plot training progress')
     parser.add_argument('--plot_env', default='main', type=str,
                         help='plot env name')
     parser.add_argument('--save', default='trained_models', type=str,
                         help='save the model after training')
     parser.add_argument('--save_every', default=100, type=int,
                         help='save the model after every n_th epoch')
     parser.add_argument('--load', default='trained_models', type=str,
                         help='load the model')
     parser.add_argument('--display', action="store_true", default=False,
                         help='Display environment state')


     parser.add_argument('--random', action='store_true', default=False,
                         help="enable random model")

     # CommNet specific args
     parser.add_argument('--commnet', action='store_true', default=False,
                         help="enable commnet model")
     parser.add_argument('--ic3net', action='store_true', default=True,
                         help="enable commnet model")
     parser.add_argument('--nagents', type=int, default=nagents,
                         help="Number of agents (used in multiagent)")
     parser.add_argument('--comm_mode', type=str, default='avg',
                         help="Type of mode for communication tensor calculation [avg|sum]")
     parser.add_argument('--comm_passes', type=int, default=1,
                         help="Number of comm passes per step over the model")
     parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                         help="Whether communication should be there")
     parser.add_argument('--mean_ratio', default=1.0, type=float,
                         help='how much coooperative to do? 1.0 means fully cooperative')
     parser.add_argument('--rnn_type', default='MLP', type=str,
                         help='type of rnn to use. [LSTM|MLP]')
     parser.add_argument('--detach_gap', default=10, type=int,
                         help='detach hidden state and cell state for rnns at this interval.'
                         + ' Default 10000 (very high)')
     parser.add_argument('--comm_init', default='uniform', type=str,
                         help='how to initialise comm weights [uniform|zeros]')
     parser.add_argument('--hard_attn', default=False, action='store_true',
                         help='Whether to use hard attention: action - talk|silent')
     parser.add_argument('--comm_action_one', default=comm_action_one, action='store_true',
                         help='Whether to always talk, sanity check for hard attention.')
     parser.add_argument('--comm_action_zero', default=comm_action_zero, action='store_true',
                         help='Whether to never talk.')
     parser.add_argument('--advantages_per_action', default=False, action='store_true',
                         help='Whether to multipy log porb for each chosen action with advantages')
     parser.add_argument('--share_weights', default=False, action='store_true',
                         help='Share weights for hops')
     parser.add_argument('--log_dir', default='tb_logs', type=str,
                         help='directory to save tensorboard logs')
     parser.add_argument('--exp_name', default=exp_name, type=str,
                         help='directory to save tensorboard logs')

     # TODO: Sanity check so as to make sure discrete and proto works for environments other than predator-prey.
     #  Currently the discrete and prototype based methods will only really take effect from inside the CommNet.
     parser.add_argument('--use_proto', default=False, action='store_true',
                         help='Whether to use prototype nets in the communication layer.')

     parser.add_argument('--discrete_comm', default=discrete_comm, action='store_true',
                         help='Whether to use discrete_comm')
     parser.add_argument('--num_proto', type=int, default=num_proto,
                         help="Number of prototypes to use")
     parser.add_argument('--add_comm_noise', default=False, action='store_true',
                         help='Whether to add noise to communication')

     parser.add_argument('--comm_dim', type=int, default=comm_dim,
                         help="Dimension of the communication vector")


     # TODO: Formalise this gating head penalty factor
     parser.add_argument('--gating_head_cost_factor', type=float, default=gating_head_cost_factor,
                         help='discount factor')
     parser.add_argument('--restore', action='store_true', default=False,
                         help='plot training progress')

     # gating reward curriculum
     parser.add_argument('--gate_reward_curriculum', action='store_true', default=reward_curriculum,
                         help='use gated reward curriculum')

     # open gate / variable gate curriculum
     parser.add_argument('--variable_gate', action='store_true', default=variable_gate,
                         help='use variable gate curriculum')
     # optimizer
     parser.add_argument('--optim_name', default='RMSprop', type=str,
                         help='pytorch optimizer')
     # learning rate scheduler
     parser.add_argument('--scheduleLR', action='store_true', default=False,
                         help='Cyclic learning rate scheduler')



     # communication maximum budget
     parser.add_argument('--budget', type=float, default=1.0,
                         help='Communication budget')

     parser.add_argument('--use_tj_curric', action='store_true', default=False,
                         help='Use curric for TJ')

     parser.add_argument('--soft_budget', type=float, default=1.0,
                         help='Soft comm budget')

     # use a pretrained network
     parser.add_argument('--load_pretrain', action='store_true', default=False,
                         help='load old model as pretrain')
     parser.add_argument('--pretrain_exp_name', type=str,
                         help='pretrain model name')
     
     parser.add_argument('--dim', type=int, default=dim,
                         help="Dimension of box (i.e length of road) ")
     parser.add_argument('--vision', type=int, default=vision,
                    help="Vision of car")
     parser.add_argument('--add_rate_min', type=float, default=add_rate_min,
                    help="rate at which to add car (till curr. start)")
     parser.add_argument('--add_rate_max', type=float, default=add_rate_max,
                    help=" max rate at which to add car")
     # env.add_argument('--curr_start', type=float, default=0,
     #                  help="start making harder after this many epochs [0]")
     # env.add_argument('--curr_end', type=float, default=0,
     #                  help="when to make the game hardest [0]")
     parser.add_argument('--difficulty', type=str, default=difficulty,
                    help="Difficulty level, easy|medium|hard|longer_easy")
     parser.add_argument('--vocab_type', type=str, default='bool',
                    help="Type of location vector to use, bool|scalar")

     # updated curriculum parameters
     parser.add_argument('--curr_start_epoch', type=float, default=-1.,
                    help="start making harder after this many epochs [0]")
     parser.add_argument('--curr_epochs', type=float, default=1000.,
                    help="Number of epochs of curriculum for when to make the game hardest [0]")

     return parser
