## tsfServer.py
##

"""
A WebServer for serving the web page and web socket connection in order to play
TSF through a web browser.
"""

# import pyTSF as tsf
# import agents
import sys
import time
import signal
import argparse
from argparse import ArgumentParser
import time, os

import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from evaluator import Evaluator
from args import get_args
from inspect import getfullargspec
from action_utils import *

import string
import random
import logging
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os.path
import uuid
import time
import json
from tornado.options import define, options
import datetime
import dateutil.tz

LOG_ROOT = os.path.abspath("data/tj")

define("port", default=8005, help="run on the given port", type=int)

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                         'reward', 'misc'))

class TSFServerProxy():
    """
    A class to act as a proxy listener for an instance of the web application
    """

    def __init__(self, websocket_handler):
        """
        Create the proxy listener
        """

        self.websocket_handler = websocket_handler

        # NECESSARY: Need to call the GameListener __init__ in order to be
        # considered a GameListener in the C++ code
        # tsf.GameListener.__init__(self)

    def notify(self, gameState):
        """
        """

        self.websocket_handler.on_game_state(gameState)

    def gameOver(self):
        """
        """

        self.websocket_handler.on_game_over()


class TSFWebApplication(tornado.web.Application):
    """
    The main class to serve the TSF web application
    """

    def __init__(self):
        """
        """

        handlers = [(r"/", MainHandler),
                    (r"/login", LoginHandler),
                    (r"/logout", LogoutHandler),
                    (r"/tsfsocket", TSFWebSocketHandler)]

        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "web/templetes"),
            static_path=os.path.join(os.path.dirname(__file__), "web/static"),
            xsrf_cookies=True,
        )
        super(TSFWebApplication, self).__init__(handlers, **settings)


class BaseHandler(tornado.web.RequestHandler):
    """
    Common functionality for all handlers
    """

    def get_current_user(self):
        return self.get_secure_cookie("user")


class MainHandler(BaseHandler):
    """
    The handler for serving the main html page
    """

    def get(self):
        """
        Send index.html
        """

        # If there isn't a logged in user, redirect to the login page
        if not self.current_user:
            print("User Not Logged In")
            self.redirect("/login")
            return
        else:
            print("User Logged In:", self.current_user)

        # Otherwise, redirect to the game
        self.render("traffic_junction.html", error=None)


class LoginHandler(BaseHandler):
    """
    The handler for letting the user login
    """

    def get(self):
        """
        Send the login page
        """

        self.render("login_tj.html", error=None)

    def post(self):
        """
        Handle reception of a post from the login page.  The post should contain
        an arument "name", which contains the name of the user.
        """

        user_name = self.get_argument("name")

        # If no username is provided, then redirect to the login page
        # TODO:  Add to the template an indication that no user name was provided
        if user_name is None or user_name == "":
            self.render("login_tj.html", error="No Username Provided")
            return

        self.set_secure_cookie("user", self.get_argument("name"))
        self.redirect("/")


class LogoutHandler(BaseHandler):
    """
    The handler for letting the user log out
    """

    def get(self):
        """
        Clear out the login cookie and redirect to the root of the application
        """

        self.clear_cookie("user")
        self.redirect("/")


class TSFWebSocketHandler(tornado.websocket.WebSocketHandler):
    """
    A class for handling bidirectional communication between the browser client
    and an instance of TSF running on the server
    """

    def get_compression_options(self):
        # Non-None enables compression with default options
        return {}

    def gameHandler(self):
        """
        """

        # Send the player command
        self.player.command(self.playerCommand)
        self.playerCommand.fire = False
        self.agent.update()

        # Tick the clock
        self.game.gameClock.tick()

    def open(self):
        """

        """

        self.username = self.get_secure_cookie("user").decode()



        print('Game start')
        self.numTrial = 40
        self.best = 20
        self.currentTrial = 1

        self.gameHandlerCallback = tornado.ioloop.PeriodicCallback(self.gameHandler, 1)

        # Player command to be updated as messages are received
        self.playerCommand = None


        # self.condition = np.random.randint(3)
        self.condition = 0
        self.n_correct_loc_preds = 0
        self.total_loc_preds = 0
        self.total_autoencoder_loss = 0
        self.n_loss_checks = 0

    def on_close(self):
        """
        The WebSocket is closed.  Logger results need to be saved.
        """

        # Perform a Game Over in order to close out, e.g., logger
        if hasattr(self, 'gameState'):
            self.on_game_over()

        # Kill the agent thread
        #        self.agent.stop()
        #        self.agent.kill()
        self.gameHandlerCallback.stop()
        # Get rid of the instance of builder and game
        # del self.logger

    #        del self.agent


    def load(self,args,path):
        # d = torch.load(path)
        # policy_net.load_state_dict(d['policy_net'])
        load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
        print(f"load directory is {load_path}")
        log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
        print(f"log dir directory is {log_path}")
        save_path = load_path

        if 'best_model.pt' in os.listdir(load_path):
            print(load_path)
            model_path = os.path.join(load_path, "best_model.pt")
        elif 'model.pt' in os.listdir(load_path):
            print(load_path)
            model_path = os.path.join(load_path, "model.pt")
        else:
            all_models = sort([int(f.split('.pt')[0]) for f in os.listdir(load_path)])
            model_path = os.path.join(load_path, f"{all_models[-1]}.pt")

        d = torch.load(model_path)
        self.policy_net.load_state_dict(d['policy_net'],strict=False)

    def on_message(self, message):
        # Try parsing as json
        try:
            message_json = json.loads(message)
        except:
            # Not JSON, so just return
            return

        if message_json["type"] == "information" and message_json["message"] == "start":
            torch.utils.backcompat.broadcast_warning.enabled = True
            torch.utils.backcompat.keepdim_warning.enabled = True
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.t = 0


            parser = get_args()
            init_args_for_env(parser)
            args = parser.parse_args()

            if args.ic3net:
                args.commnet = 1
                args.hard_attn = 1
                args.mean_ratio = 0

                # For TJ set comm action to 1 as specified in paper to showcase
                # importance of individual rewards even in cooperative games
                # if args.env_name == "traffic_junction":
                #     args.comm_action_one = True
            # Enemy comm
            args.nfriendly = args.nagents
            if hasattr(args, 'enemy_comm') and args.enemy_comm:
                if hasattr(args, 'nenemies'):
                    args.nagents += args.nenemies
                else:
                    raise RuntimeError("Env. needs to pass argument 'nenemy'.")

            self.env = data.init(args.env_name, args, False)

            num_inputs = self.env.observation_dim
            args.num_actions = self.env.num_actions

            # Multi-action
            if not isinstance(args.num_actions, (list, tuple)): # single action case
                args.num_actions = [args.num_actions]
            args.dim_actions = self.env.dim_actions
            args.num_inputs = num_inputs

            # Hard attention
            if args.hard_attn and args.commnet:
                # add comm_action as last dim in actions
                args.num_actions = [*args.num_actions, 2]
                args.dim_actions = self.env.dim_actions + 1

            # Recurrence
            if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
                args.recurrent = True
                args.rnn_type = 'LSTM'


            parse_action_args(args)

            if args.seed == -1:
                args.seed = np.random.randint(0,10000)
            torch.manual_seed(args.seed)

            print(args)
            print(args.seed)

            if args.commnet:
                self.policy_net = CommNetMLP(args, num_inputs, train_mode=False)
            elif args.random:
                self.policy_net = Random(args, num_inputs)

            # this is what we are working with for IC3 Net predator prey.
            elif args.recurrent:
                self.policy_net = RNN(args, num_inputs)
            else:
                self.policy_net = MLP(args, num_inputs)

            self.load(args, "")
            self.policy_net.eval()
            if not args.display:
                display_models([self.policy_net])

            # share parameters among threads, but not gradients
            for p in self.policy_net.parameters():
                p.data.share_memory_()


            self.args = args


            self.policy_net.num_null = 0
            self.policy_net.num_good_comms = 0
            self.policy_net.num_cut_comms = 0
            self.policy_net.num_comms = 0


            self.all_comms = []
            self.episode = []
            epoch = 1
            reset_args = getfullargspec(self.env.reset).args
            if 'epoch' in reset_args:
                self.state = self.env.reset(epoch)
            else:
                self.state = self.env.reset()

            self.total_autoencoder_loss = 0
            should_display = False

            # if should_display:
            #    self.env.display()
            self.stat = dict()
            self.info = dict()
            switch_t = -1

            self.prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

            self.done = False
            self.step = 0
            self.history = []
            self.actionList = []
            self.moveRT = []
            self.results = 'other'

            while not self.env.get_alive_wrapper(0):
                if self.step > 20 or self.done:
                    if 'epoch' in reset_args:
                        self.state = self.env.reset(epoch)
                    else:
                        self.state = self.env.reset()
                    self.step = 0
                    print ("resetting step..")
                should_display = False
                t = self.t
                misc = dict()
                if t == 0 and self.args.hard_attn and self.args.commnet:
                    self.info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                    self.info['step_t'] = t  #

                self.info['record_comms'] = 99

                if self.args.recurrent:
                    if self.args.rnn_type == 'LSTM' and t == 0:
                        self.prev_hid = self.policy_net.init_hidden(batch_size=self.state.shape[0])

                    x = [self.state, self.prev_hid]
                    action_out, value, self.prev_hid, proto_comms = self.policy_net(x, self.info)

                    if (t + 1) % self.args.detach_gap == 0:
                        if self.args.rnn_type == 'LSTM':
                            self.prev_hid = (self.prev_hid[0].detach(), self.prev_hid[1].detach())
                        else:
                            self.prev_hid = self.prev_hid.detach()
                else:
                    x = self.state
                    action_out, value, filtered_comms = self.policy_net(x, self.info)

                action = select_action(self.args, action_out, eval_mode=True)
                action, actual = translate_action(self.args, self.env, action)
                next_state, reward, done, info = self.env.step(actual)


                self.collisionOrNot, self.collisionLocation = self.env.get_reached_wrapper()
                self.done = done or self.step >= self.args.max_steps or bool(self.collisionOrNot)

                episode_mask = np.ones(reward.shape)
                episode_mini_mask = np.ones(reward.shape)

                if self.done:
                    episode_mask = np.zeros(reward.shape)
                else:
                    if 'is_completed' in info:
                        episode_mini_mask = 1 - info['is_completed'].reshape(-1)

                if should_display:
                    self.env.display()

                trans = Transition(self.state, action, action_out, value, episode_mask, episode_mini_mask, next_state,
                                   reward, misc)
                self.episode.append(trans)
                self.state = next_state
                self.t = t + 1
                self.info = info


            if t == 0 and self.args.hard_attn and self.args.commnet:
                self.info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                self.info['step_t'] = t  #

            self.info['record_comms'] = 99

            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    self.prev_hid = self.policy_net.init_hidden(batch_size=self.state.shape[0])

                x = [self.state, self.prev_hid]
                self.action_out, self.value, self.prev_hid, filtered_comms = self.policy_net(x, self.info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        self.prev_hid = (self.prev_hid[0].detach(), self.prev_hid[1].detach())
                    else:
                        self.prev_hid = self.prev_hid.detach()
            else:
                x = self.state
                self.action_out, self.value, filtered_comms = self.policy_net(x, self.info)

            self.human_path = self.env.get_path_wrapper(0)
            self.all_other_paths = [self.env.get_path_wrapper(i).tolist() if type(self.env.get_path_wrapper(i)) != int else self.env.get_path_wrapper(i) for i in range(1,self.args.nagents)]

            self.initialStep = self.step
            self.car_loc = self.env.get_loc_wrapper()


            self.message = {}
            #==============================================================================================================
            if self.args.autoencoder:
                decoded = self.policy_net.decode().detach()
                decoded = decoded.squeeze()
                player_decoded = decoded[0]

                player_decoded_locs = player_decoded[1:,2:-2]
                player_decoded_actions = player_decoded[1:,-1]
                player_decoded_actions = np.where(player_decoded_actions > 1, 1, 0)


                player_decoded_locs = torch.nn.functional.softmax(player_decoded_locs/0.0001,dim=1).tolist()

                pred_loc_i = 0
                for i in range(1,self.args.nagents):
                    if self.env.get_alive_wrapper(i):

                        player_decoded_locs[i-1] = np.round(player_decoded_locs[i-1],0)
                        pred_loc = self.env.obs2pos_wrapper(tuple(player_decoded_locs[i-1]))
                        if pred_loc is None:
                            print ("ERROR DECODING PRED LOC")
                            pred_loc = [0,0]
                        if self.args.autoencoder_action:
                            pred_action = player_decoded_actions[i-1]
                            self.message[i] = {"loc": [[pred_loc[0],pred_loc[1]]],"action":pred_action.tolist(),"conf": 1}
                        else:
                            self.message[i] = {"loc": [[pred_loc[0],pred_loc[1]]], "conf": 1}
                    else:
                        self.message[i] = {}

            else:
                self.message[i] = {}
            #==============================================================================================================


            # print(self.car_loc)
            self.history.append(self.car_loc.tolist())
            self.gameState = {
                'players': self.car_loc.tolist(),
                'humanPath': self.human_path.tolist(),
                'comm': None,
                'message': self.message,
                "contains_action":self.args.autoencoder_action,
                "otherPaths":self.all_other_paths,
                'step': self.step,
                'done': self.done,
                'currentTrial': self.currentTrial,
                'history': self.history,
                'condition':self.condition
            }

            self.gameStateJson = json.dumps(self.gameState)
            self.startTimer = time.time()

            if self.currentTrial == 12:
                self.write_message({'attentionCheck':True})
                return
            self.write_message(self.gameStateJson)

            return

        if message_json["type"] == "command":


            self.all_other_paths = [self.env.get_path_wrapper(i).tolist() if type(self.env.get_path_wrapper(i)) != int else self.env.get_path_wrapper(i) for i in range(1,self.args.nagents)]
            # Pull out the command
            command = message_json["message"]
            if self.currentTrial == 12:
                self.gameState = {
                    'players': None,
                    'humanPath': None,
                    'comm': None,
                    'message': None,
                    "contains_action":self.args.autoencoder_action,
                    "otherPaths":self.all_other_paths,
                    'moveRT': None,
                    'step': None,
                    'best': None,
                    'done': None,
                    'results': command,
                    'collisionLocation': None,
                    'currentTrial': None,
                    'history': None,
                    'condition':self.condition
                }
                self.currentTrial += 1
                self.save_log()

                self.gameState = {
                    'players': self.car_loc.tolist(),
                    'humanPath': self.human_path.tolist(),
                    'comm': None,
                    'message': self.message,
                    "contains_action":self.args.autoencoder_action,
                    "otherPaths":self.all_other_paths,
                    'step': self.step,
                    'done': self.done,
                    'currentTrial': self.currentTrial,
                    'history': self.history,
                    'condition': self.condition
                }
                self.startTimer = time.time()

                self.gameStateJson = json.dumps(self.gameState)
                self.write_message(self.gameStateJson)
                return

            if self.step != self.initialStep and self.done:
                print('reject invalid action')
                return

            if command["command"] == "go":
                self.humanAction = 0
            elif command["command"] == "brake":
                self.humanAction = 1
            else:
                return
            self.actionList.append(self.humanAction)
            if self.step == self.initialStep:
                self.moveRT.append(time.time() - self.startTimer)
                self.lastMove = time.time()
            else:
                self.moveRT.append(time.time() - self.lastMove)
                self.lastMove = time.time()
            self.info['replace_comm'] = False
            self.step += 1
            should_display = False
            t = self.t
            misc = dict()
            action = select_action(self.args, self.action_out)
            action, actual = translate_action(self.args, self.env, action)
            # actual[0][0] = self.humanAction
            next_state, reward, done, info = self.env.step(actual)

            self.collisionOrNot, self.collisionLocation = self.env.get_reached_wrapper()
            # self.done = done or self.step >= self.args.max_steps or bool(self.collisionOrNot) or not bool(self.env.get_alive_wrapper(0))
            self.reached = (self.human_path[-1][0] == self.car_loc[0][0]) and (self.human_path[-1][1] == self.car_loc[0][1])
            self.done = done or self.step >= self.args.max_steps or bool(self.collisionOrNot) or bool(self.reached)

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if self.done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)


            trans = Transition(self.state, action, self.action_out, self.value, episode_mask, episode_mini_mask, next_state,
                               reward, misc)
            self.episode.append(trans)
            self.state = next_state
            self.t = t + 1
            self.info = info

            if bool(self.collisionOrNot):
                self.results = 'collision'
            elif bool(self.reached):
                self.results = 'success'
            elif self.step >= self.args.max_steps:
                self.results = 'timeout'
            else:
                self.results = 'other'

            if self.done:
                self.currentTrial += 1
                if self.step < self.best:
                    self.best = self.step
                self.save_log()
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    self.prev_hid = self.policy_net.init_hidden(batch_size=self.state.shape[0])

                x = [self.state, self.prev_hid]
                self.action_out, self.value, self.prev_hid, filtered_comms = self.policy_net(x, self.info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        self.prev_hid = (self.prev_hid[0].detach(), self.prev_hid[1].detach())
                    else:
                        self.prev_hid = self.prev_hid.detach()
            else:
                x = self.state
                self.action_out, self.value, filtered_comms = self.policy_net(x, self.info)

            # print(action_out)
            # print('filtered_comms', filtered_comms)
            self.message = {}

            if self.args.autoencoder:
                decoded = self.policy_net.decode().detach()

                if self.args.autoencoder and self.args.autoencoder_action:
                    x_all = x[0].expand(self.args.nagents,self.args.nagents, -1)
                    gt_actions = torch.tensor(actual[0]).unsqueeze(1).expand(self.args.nagents,self.args.nagents,-1)
                    x_all = torch.cat((x_all,gt_actions),dim=2)

                    loss_autoencoder = torch.nn.functional.mse_loss(decoded, x_all)
                    self.total_autoencoder_loss += loss_autoencoder.detach().numpy()
                    self.n_loss_checks += 1

                print ("total autoencoder loss: " + str(self.total_autoencoder_loss))

                decoded = decoded.squeeze()

                player_decoded = decoded[0]

                player_decoded_locs = player_decoded[1:,2:-2]
                player_decoded_actions = player_decoded[1:,-1]
                player_decoded_actions = np.where(player_decoded_actions >= 1, 1,0)

                player_decoded_locs = torch.nn.functional.softmax(player_decoded_locs/0.0001,dim=1).tolist()
                pred_loc_i = 0
                for i in range(1,self.args.nagents):
                    if self.env.get_alive_wrapper(i):
                        player_decoded_locs[i-1] = np.round(player_decoded_locs[i-1],0)
                        pred_loc = self.env.obs2pos_wrapper(tuple(player_decoded_locs[i-1]))
                        if pred_loc is None:
                            print (player_decoded_locs[i-1])
                            print ("ERROR DECODING PRED LOC")
                            pred_loc = [0,0]

                        if self.args.autoencoder_action:
                            pred_action = player_decoded_actions[i-1]
                            self.message[i] = {"loc": [[pred_loc[0],pred_loc[1]]],"action":pred_action.tolist(),"conf": 1}
                        else:
                            self.message[i] = {"loc": [[pred_loc[0],pred_loc[1]]], "conf": 1}
                    else:
                        self.message[i] = {}

            else:
                self.message[i] = {}

            self.car_loc = self.env.get_loc_wrapper()

            self.history.append(self.car_loc.tolist())
            self.gameState = {
                'players': self.car_loc.tolist(),
                'humanPath': self.human_path.tolist(),
                'comm': None,
                'message': self.message,
                "contains_action":self.args.autoencoder_action,
                "otherPaths":self.all_other_paths,
                'moveRT': self.moveRT,
                'step': self.step,
                'best': self.best,
                'done': self.done,
                'results': self.results,
                'collisionLocation': self.collisionLocation,
                'currentTrial': self.currentTrial,
                'history': self.history,
                'actionList': self.actionList,
                'condition': self.condition
            }

            self.gameStateJson = json.dumps(self.gameState)

            self.write_message(self.gameStateJson)

        if message_json['type']=='survey':

            self.surveyResults = {
                'name':self.username,
                "randomCode":message_json["randomCode"],
                "helpful":message_json["helpful"],
                "understand":message_json["understand"],
                "satisfy":message_json["satisfy"],
                "Post_difficulty": message_json["Post_difficulty"],
                "Post_how":message_json["Post_how"],
                "Post_feedback": message_json["Post_feedback"],
            }

            self.save_log()


    def save_log(self):

        print("log saved")

        # random.choice(string.ascii_uppercase) for _ in range(8)
        # Dump the log

        if hasattr(self, 'surveyResults'):
            log_path = os.path.join(LOG_ROOT, self.username)

            # Create the folder, if it doesn't exist
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                print("Created path: ", log_path)

            # Create a unique filename
            now = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
            if self.condition ==1:
                group = 'med'
            elif self.condition == 2:
                group = 'min'
            else:
                group = 'fixed'
            random_string = now + '_' + '_'.join([group, self.username])
            log_filename = 'survey_results_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)

            log_file_path = os.path.join(log_path, log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)

            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            with open(log_file_path, 'a+') as outfile:
                json.dump(self.surveyResults, outfile)

        if self.gameState is not None:
            log_path = os.path.join(LOG_ROOT, self.username)

            # Create the folder, if it doesn't exist
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                print("Created path: ", log_path)

            # Create a unique filename
            now = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
            if self.condition ==1:
                group = 'med'
            elif self.condition == 2:
                group = 'min'
            else:
                group = 'fixed'
            random_string = now + '_' + '_'.join([group, self.username, str(self.currentTrial)])
            log_filename = 'game_log_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)


            log_file_path = os.path.join(log_path, log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)

            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            with open(log_file_path, 'a+') as outfile:
                json.dump(self.gameState, outfile)


    def on_game_over(self):
        print("log saved")

        # random.choice(string.ascii_uppercase) for _ in range(8)
        # Dump the log
        if self.gameState is not None:
            log_path = os.path.join(LOG_ROOT, self.username)

            # Create the folder, if it doesn't exist
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                print("Created path: ", log_path)

            # Create a unique filename
            now = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
            if self.condition ==1:
                group = 'med'
            elif self.condition == 2:
                group = 'min'
            else:
                group = 'fixed'
            random_string = now + '_' + '_'.join([group, self.username, str(self.currentTrial)])
            log_filename = 'game_log_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)

            log_file_path = os.path.join(log_path, log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)

            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            with open(log_file_path, 'a+') as outfile:
                json.dump(self.gameState, outfile)


if __name__ == "__main__":
    # tornado.options.parse_command_line()
    print ("Running on port " + str(options.port))
    app = TSFWebApplication()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
