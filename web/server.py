## tsfServer.py
##

"""
A WebServer for serving the web page and web socket connection in order to play
TSF through a web browser.
"""

import pyTSF as tsf
import agents


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



class TSFServerProxy(tsf.GameListener):
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
        tsf.GameListener.__init__(self)


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
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
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
        self.agent_class_name = random.choice(list(agents.repo.keys()))
        print(self.agent_class_name)

        # agent_class = agents.repo[self.agent_class_name][0]
        # self.agent = agent_class(self.builder.getPlayer(1))
        # self.game.addListener(self.agent)
    #    self.agent.start()

        # Create a proxy listener and link it to the game
        # self.proxy_listener=TSFServerProxy(self)
        # self.game.addListener(self.proxy_listener)

        self.gameHandlerCallback = tornado.ioloop.PeriodicCallback(self.gameHandler, 1)
        
        # Player command to be updated as messages are received
        self.playerCommand = tsf.PlayerCommand()
        self.playerCommand.thrust = False
        self.playerCommand.turn = tsf.NO_TURN
        self.playerCommand.fire = False

        self.playerFired = False


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
            return


        if message_json["type"] == "command":
            # Pull out the command
            command = message_json["message"]
        
            # See if the JSON has the correct elements
            if not "player" in command or not "command" in command or not "isPress" in command:
                return

            # Create a command from the message
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
