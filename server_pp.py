## tsfServer.py
##

"""
A WebServer for serving the web page and web socket connection in order to play
TSF through a web browser.
"""

#import pyTSF as tsf
#import agents
import sys
import time
import signal
import argparse
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
from inspect import getargspec
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

LOG_ROOT = os.path.abspath("/data3/tsf_logs/")


define("port", default=8888, help="run on the given port", type=int)

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
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
        #tsf.GameListener.__init__(self)


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
        self.render("index.html", error=None)



class LoginHandler(BaseHandler):
    """
    The handler for letting the user login
    """

    def get(self):
        """
        Send the login page
        """

        self.render("login.html", error=None)


    def post(self):
        """
        Handle reception of a post from the login page.  The post should contain
        an arument "name", which contains the name of the user.
        """

        user_name = self.get_argument("name")

        # If no username is provided, then redirect to the login page
        # TODO:  Add to the template an indication that no user name was provided
        if user_name is None or user_name == "":
            self.render("login.html", error="No Username Provided")
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

        # # Create an instance of the TSF game
        # self.builder=tsf.JsonGameBuilder("configs/sf_config.json")
        # self.game=self.builder.build()
        # self.logger=tsf.Logger()
        # self.game.addListener(self.logger)

        # Create and bind a random agent
        # self.player = self.builder.getPlayer(0)
        #self.agent_class_name = random.choice(list(agents.repo.keys()))
        #print(self.agent_class_name)

        # agent_class = agents.repo[self.agent_class_name][0]
        # self.agent = agent_class(self.builder.getPlayer(1))
        # self.game.addListener(self.agent)
    #    self.agent.start()

        # Create a proxy listener and link it to the game
        # self.proxy_listener=TSFServerProxy(self)
        # self.game.addListener(self.proxy_listener)

        self.gameHandlerCallback = tornado.ioloop.PeriodicCallback(self.gameHandler, 1)
        
        # Player command to be updated as messages are received
        self.playerCommand = None


    def on_close(self):
        """
        The WebSocket is closed.  Logger results need to be saved.
        """

        # Perform a Game Over in order to close out, e.g., logger
        self.on_game_over()

        # Kill the agent thread
#        self.agent.stop()
#        self.agent.kill()
        self.gameHandlerCallback.stop()
        # Get rid of the instance of builder and game
        del self.builder
        del self.game
        del self.proxy_listener
        del self.gameHandlerCallback
        del self.logger
#        del self.agent

    def load(self, args, path):
        # d = torch.load(path)
        # policy_net.load_state_dict(d['policy_net'])
        args.seed = 7382

        load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
        print(f"load directory is {load_path}")
        log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
        print(f"log dir directory is {log_path}")
        save_path = load_path

        if 'model.pt' in os.listdir(load_path):
            print(load_path)
            model_path = os.path.join(load_path, "model.pt")

        else:
            all_models = sort([int(f.split('.pt')[0]) for f in os.listdir(load_path)])
            model_path = os.path.join(load_path, f"{all_models[-1]}.pt")

        d = torch.load(model_path)
        self.policy_net.load_state_dict(d['policy_net'])

    def on_message(self, message):
        """
        """


#        if message=="start":
#            self.game.start()
#            self.gameHandlerCallback.start()
#            return 

        # Try parsing as json
        try:
            message_json = json.loads(message)
        except:
            # Not JSON, so just return
            return


        if message_json["type"] == "information" and message_json["message"] == "start":
            '''
            # Create an instance of the TSF game
            self.builder=tsf.JsonGameBuilder("configs/sf_config.json")
            self.game=self.builder.build()
            self.logger=tsf.Logger()
            self.game.addListener(self.logger)
            self.player = self.builder.getPlayer(0)

            agent_class = agents.repo[self.agent_class_name][0]
            self.agent = agent_class(self.builder.getPlayer(1))
            self.game.addListener(self.agent)
            
            # Create a proxy listener and link it to the game
            self.proxy_listener=TSFServerProxy(self)
            self.game.addListener(self.proxy_listener)

            self.game.reset()
            self.logger.reset()
            self.game.start()
            self.gameHandlerCallback.start()
            '''
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

            self.load(args, args.load)

            if not args.display:
                display_models([self.policy_net])

            # share parameters among threads, but not gradients
            for p in self.policy_net.parameters():
                p.data.share_memory_()
            
            self.args = args
            
            self.all_comms = []
            self.episode = []
            epoch = 1
            reset_args = getargspec(self.env.reset).args
            if 'epoch' in reset_args:
                self.state = self.env.reset(epoch)
            else:
                self.state = self.env.reset()
            should_display = False

            if should_display:
                self.env.display()
            self.stat = dict()
            self.info = dict()
            switch_t = -1

            self.prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
            return


        if message_json["type"] == "command":
            # Pull out the command
            command = message_json["message"]
        
            # See if the JSON has the correct elements
            if not "player" in command or not "command" in command or not "isPress" in command:
                return

            # Create a command from the message
            '''
            self.playerCommand.fire = False

            if command["command"] == "thrust":
                self.playerCommand.thrust = command["isPress"]
            elif command["command"] == "turn_left":
                self.playerCommand.turn = tsf.TURN_LEFT if command["isPress"] else tsf.NO_TURN
            elif command["command"] == "turn_right":
                self.playerCommand.turn=tsf.TURN_RIGHT if command["isPress"] else tsf.NO_TURN

            if command["command"] == "fire": 
                if command["isPress"] and not self.playerFired:
                    self.playerCommand.fire = True
                    self.playerFired = True            
                if not command["isPress"]:
                    self.playerFired = False
            '''
            should_display = False
            t = self.t
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                self.info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    self.prev_hid = self.policy_net.init_hidden(batch_size=self.state.shape[0])

                x = [self.state, self.prev_hid]
                action_out, value, self.prev_hid = self.policy_net(x, self.info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        self.prev_hid = (self.prev_hid[0].detach(), self.prev_hid[1].detach())
                    else:
                        self.prev_hid = self.prev_hid.detach()
            else:
                x = self.state
                action_out, value = self.policy_net(x, self.info)
            
            #print(action_out)

            action = select_action(self.args, action_out)
            #print(action)
            action, actual = translate_action(self.args, self.env, action)
            #print(actual)
            next_state, reward, done, info = self.env.step(actual)
            #print(next_state)
            print(self.env.get_pp_loc_wrapper())

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                # print("before ", stat.get('comm_action', 0), info['comm_action'][:self.args.nfriendly])
                self.stat['comm_action'] = self.stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                self.all_comms.append(info['comm_action'][:self.args.nfriendly])
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    self.stat['enemy_comm']  = self.stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            self.stat['reward'] = self.stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                self.stat['enemy_reward'] = self.stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

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

            trans = Transition(self.state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            self.episode.append(trans)
            self.state = next_state
            self.t = t + 1
            #print(t)
            #print(self.t)
        

    def on_game_state(self, gameState):
        """
        Callback from a proxy game listener
        """


        # HACK:  Add the time information to the json before converting to
        # a string.  Should probably have this in the C++ code...

        gameStateJson = json.loads(gameState.toJsonString())

        gameStateJson["time"] = self.game.gameClock.getTime()
        gameStateJson["tick"] = self.game.gameClock.getTick()        

        gameStateJson = json.dumps(gameStateJson)

        self.write_message(gameStateJson)        


    def on_game_over(self):
        """
        Callback from a proxy game listener
        """

        print("on_game_over")

        self.game.stop()
       
 
        # Dump the log
        if self.logger is not None:
            log_path = os.path.join(LOG_ROOT,self.username,self.agent_class_name)

            # Create the folder, if it doesn't exist
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                print("Created path: ", log_path)
            
            # Create a unique filename
            now = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
            random_string = now + ''.join(random.choice(string.ascii_uppercase) for _ in range(8))
            log_filename = 'game_log_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)

            while log_filename in existing_files:
                print("Creating new log_filename")
                random_string = now + ''.join(random.choice(string.ascii_uppercase) for _ in range(8))
                log_filename = 'game_log_%s.json' % random_string
                # metadata_filename = 'game_log_%s.meta' % random_string



            log_file_path = os.path.join(log_path,log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)
 
            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            self.logger.dump(log_file_path)

            # with open(meta_file_path,'w') as meta_file:
            #     meta_file.write('Agent Class: %s\n' % self.agent_class_name)



if __name__=="__main__":
    tornado.options.parse_command_line()
    app = TSFWebApplication()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
