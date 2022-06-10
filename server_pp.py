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
import time, os

import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from evaluator import Evaluator
from args_pp import get_args
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

LOG_ROOT = os.path.abspath("/home/huaol/data/0129randomonehot")

define("port", default=8002, help="run on the given port", type=int)

Transition = namedtuple('Transition',
                        ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                         'reward', 'misc'))

token_dict = [{
    1:{
        'raw':[0.23801799161900206, 0.7325141653996786, 0.19846114195194223, 0.5393944003862026, 0.4613124880046595, 0.10604421127378937, 0.3634451356342147, 0.6636483076769099, 0.8419504100027456],
        'pca':[0.15182869,-0.47542515],
        'loc':[3.,1.]
    },
    2: {
        'raw': [0.3078763153536957, 0.14805834950618751, 0.3097069760365357, 0.8370009988023058, 0.8478655833306199, 0.29897639440391244, 0.3326006632661294, 0.7599369781457616, 0.8825214374555808],
        'pca': [-0.24235778,-0.69981944],
        'loc': [1.61111111,0.38888889]
    },
    3: {
        'raw': [0.8819358330857345, 0.2627080195125338, 0.62647271479697, 0.169748378569708, 0.34497638629894034, 0.7300588380985256, 0.3317179862594988, 0.17439549365924764, 0.08449755976323294],
        'pca': [-0.0904792,0.75396827],
        'loc': [3.73684211,7.05263158]
    },
    4: {
        'raw': [0.20142512309713748, 0.1473080108678259, 0.3277002287950114, 0.875473700525893, 0.7963215795273852, 0.8478443242888444, 0.07495478460179618, 0.8744399197670089, 0.8360873788653396],
        'pca': [-0.5098786,-0.63763875],
        'loc': [1.58536585,1.]
    },
    5: {
        'raw': [0.08955887915571893, 0.27272336253766816, 0.924240663064384, 0.12190755582969964, 0.0860216133690335, 0.058724524162691374, 0.8617009143190693, 0.7219024078162108, 0.27013556266760846],
        'pca': [1.01246488,0.02155013],
        'loc': [7.34042553,3.55319149]
    },
    6: {
        'raw': [0.6870517842572824, 0.7539627670929671, 0.5160086262278977, 0.7881142784163332, 0.7153026644021444, 0.8819773797750566, 0.12299347503232387, 0.11216835888901917, 0.09040002085285437],
        'pca': [-0.5853829,0.54071572],
        'loc': [2.18867925,6.86792453]
    },
    7: {
        'raw': [0.8394953253077313, 0.8100684305765559, 0.9273408791409841, 0.28930216689938654, 0.15808560705240937, 0.6720555770875383, 0.8413709782204191, 0.38769766639247577, 0.3186048531745279],
        'pca': [0.44402268,0.7313564],
        'loc': [6.41176471,6.57647059]
    },
    8: {
        'raw': [0.8221137435945136, 0.07835728277577594, 0.11865925580692684, 0.8615342115135893, 0.9297224013522408, 0.7893646909850021, 0.1813333902844934, 0.1211313242500813, 0.39521871371183176],
        'pca': [-0.91406448,0.12871821],
        'loc': [0.7826087,3.79347826]
    },
    9: {
        'raw': [0.07422709275573385, 0.3387281696740808, 0.824552561563955, 0.6714906195213651, 0.12512868302052751, 0.10874550678310316, 0.7718508278402992, 0.7991284059181146, 0.6097104212735481],
        'pca': [0.73384669,-0.36342538],
        'loc': [5.65254237,1.52542373]
    },
    10: {
        'raw': [0.6093685361352408, 0.48378074925892295, 0.9301173165918591, 0.10678958236225718, 0.23254922156143712, 0.1226728540104421, 0.8425635664122217, 0.68843780288164, 0.10518988427018998],
        'pca': [0.79654476,0.40897997],
        'loc': [4.04,3.778]
    },
},
{
    1:{
        'raw':[0.8830476085698017, 0.2421770048820867, 0.6417319436588336, 0.7960443566264426, 0.2560441294018934, 0.9030344461748867, 0.21696497854135846, 0.14097354536149337, 0.4936403704122023],
        'pca':[-0.39471949,-0.50866918],
        'loc':[6.55555556,1.44444444]
    },
    2: {
        'raw': [0.14351402705422842, 0.33030939212406746, 0.9158078077360784, 0.20186693786993512, 0.08784475933154115, 0.39638027376317264, 0.10073004492581149, 0.2141968941508983, 0.3238668686814056],
        'pca': [0.43181636,-0.64323601],
        'loc': [2.65217391,0.73913043]
    },
    3: {
        'raw': [0.8444627266221769, 0.0919314990998223, 0.8385295383724413, 0.4244050562509851, 0.6680379551811706, 0.7372323669545578, 0.8315077829963542, 0.31143542664397816, 0.8506959655708524],
        'pca': [-0.76526173,-0.23805912],
        'loc': [7.35714286,1.67857143]
    },
    4: {
        'raw': [0.08947565256885831, 0.9069627920918946, 0.20745037083098045, 0.8849682371424255, 0.8591534979365013, 0.2445128072996251, 0.8230117878933763, 0.9304251558843827, 0.15647307492976525],
        'pca': [0.39995019,0.97890259],
        'loc': [2.48,7.32]
    },
    5: {
        'raw': [0.8577679514422842, 0.5696885398531044, 0.8348755896194836, 0.2535510717104772, 0.21588261670756928, 0.43734324103790445, 0.12057228767419578, 0.20314668904317448, 0.873832064167431],
        'pca': [-0.13134187,-0.73184064],
        'loc': [4.8,1.3]
    },
    6: {
        'raw': [0.10493058027762386, 0.8761699068673027, 0.7240745417815555, 0.1707159933721597, 0.18753930380892236, 0.1621771545542456, 0.16518433081631256, 0.4860069542340781, 0.0975996268469547],
        'pca': [0.88274901,-0.2594784],
        'loc': [0.98412698,1.74603175]
    },
    7: {
        'raw': [0.7977907658378769, 0.33214199992442184, 0.1613125188022111, 0.8825094834379714, 0.8691995715878833, 0.6570574363334462, 0.7589085508020058, 0.2786803307522193, 0.920358103433871],
        'pca': [-0.79918901,0.35470868],
        'loc': [6.91891892,4.43243243]
    },
    8: {
        'raw': [0.18500526787193358, 0.8743911010559485, 0.13941338954862437, 0.5224037607966995, 0.347082142681408, 0.3300542984455445, 0.12070992700266334, 0.8719318400334788, 0.0860307781596471],
        'pca': [0.7912972,0.29404954],
        'loc': [1.58333333,5.21296296]
    },
    9: {
        'raw': [0.16560419626983203, 0.15925389596909142, 0.16403208089378207, 0.8895494525467008, 0.8007711159187556, 0.1951028363748561, 0.9560200253119149, 0.5620622331016353, 0.859532681074243],
        'pca': [-0.41530065,0.75362254],
        'loc': [4.11692845,4.48516579]
    }
},
{
    1:{
        'raw':[0.8053961119012601, 0.25729302003867893, 0.8762253072816232, 0.9023988640721541, 0.8368612989456136, 0.9133370748126364, 0.0974823200200066, 0.4537469801132665, 0.14927134672968326],
        'pca':[-0.76014528,0.28309524],
        'loc':[3.5,0. ]
    },
    2: {
        'raw': [0.07461749228339119, 0.35591053760529723, 0.13528010075129215, 0.08062908701781177, 0.6682304478400094, 0.09773669783044012, 0.8409811496735111, 0.8129044844919217, 0.6416479326082483],
        'pca': [0.81393815,-0.27362899],
        'loc': [5.64285714,7.14285714]
    },
    3: {
        'raw': [0.6432915827468368, 0.8007424422909014, 0.6996667523815047, 0.8457660417210343, 0.645850233527049, 0.08444037456074503, 0.8294484610899422, 0.29901674725272487, 0.5524299062014388],
        'pca': [-0.25650759,-0.49745751],
        'loc': [2.36842105,5.63157895]
    },
    4: {
        'raw': [0.748950407064908, 0.5739301687314147, 0.21433903171325788, 0.3962942134535594, 0.4070126438112658, 0.8977213254323109, 0.4491114108284965, 0.159965066853338, 0.7219985054940758],
        'pca': [-0.19725823,0.51623918],
        'loc': [4.86363636,0.54545455]
    },
    5: {
        'raw': [0.8243721187870915, 0.7202507239674864, 0.7142623808095644, 0.7044009648854183, 0.9220867202300865, 0.2865837643492861, 0.6227624972701177, 0.7445400911849646, 0.7526572629607332],
        'pca': [-0.37416876,-0.56872765],
        'loc': [1.85416667,5.6875]
    },
    6: {
        'raw': [0.15135057911168331, 0.09398970380640383, 0.10938253587228085, 0.3492038202839343, 0.06905316352156614, 0.7765433361275056, 0.41277652854807106, 0.09739797760893158, 0.181875013886112],
        'pca': [0.41373047,0.89430826],
        'loc': [5.6, 1.76363636]
    },
    7: {
        'raw': [0.31984023185924115, 0.438463202060933, 0.8887325742913778, 0.4999649121998074, 0.7344330202454433, 0.09669990523226203, 0.9034610243824901, 0.5418115251147537, 0.18499075524410286],
        'pca': [0.17507101,-0.49492079],
        'loc': [3.79365079,6.6984127]
    },
    8: {
        'raw': [0.06706793847086868, 0.07687954196229116, 0.26943413685675693, 0.06422945764501409, 0.2054911025544022, 0.22887298477137177, 0.8764949009696171, 0.8392570523989308, 0.5071762126742125],
        'pca': [0.96753181,0.01505732],
        'loc': [7.03389831,4.51694915]
    },
    9: {
        'raw': [0.7765514482190591, 0.6002246016492883, 0.7352072691032661, 0.857263263663709, 0.7329679658095458, 0.6458265360531885, 0.14539781496950963, 0.10610263769113008, 0.8909690246141759],
        'pca': [-0.78219158,0.12603493],
        'loc': [1.16176471,2.61029412]
    },
    10: {
        'raw': [0.6503524371489842, 0.42378374154551995, 0.11844582046635745, 0.784865927252023, 0.8698959205983262, 0.15769731757945618, 0.8881475067855843, 0.5730261105125649, 0.12909819960951172],
        'pca': [0.10696453,-0.29819107],
        'loc': [3.89920949,4.]
    },
}]
one_hot_dict=[{
    1:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc':[3.5,0. ]},
    2:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [5.64285714,7.14285714]},
    3:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [2.36842105,5.63157895]},
    4:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [4.86363636,0.54545455]},
    5:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [1.85416667,5.6875]},
    6:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [5.6, 1.76363636]},
    7:{'raw':[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [3.79365079,6.6984127]},
    8:{'raw':[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],'loc': [7.03389831,4.51694915]},
    9: {
        'raw': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.],
        'loc': [1.16176471,2.61029412]
    },
    10: {
        'raw': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1.],
        'loc': [3.89920949,4.]
    }
}]

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
        self.render("parent_child.html", error=None)


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
        # self.agent_class_name = random.choice(list(agents.repo.keys()))
        # print(self.agent_class_name)

        # agent_class = agents.repo[self.agent_class_name][0]
        # self.agent = agent_class(self.builder.getPlayer(1))
        # self.game.addListener(self.agent)
        #    self.agent.start()

        # Create a proxy listener and link it to the game
        # self.proxy_listener=TSFServerProxy(self)
        # self.game.addListener(self.proxy_listener)

        print('Game start')
        self.numTrial = 40
        self.best = 20
        self.currentTrial = 1
        sessionList = ['parent', 'child']
        # decide first session
        i = np.random.randint(2)
        # i = 0
        self.firstSession = sessionList[i]
        self.secondSession = sessionList[i - 1]

        self.gameHandlerCallback = tornado.ioloop.PeriodicCallback(self.gameHandler, 1)

        # Player command to be updated as messages are received
        self.playerCommand = None

        model_conditions = ['proto_fixed1', 'one_hot81']
        self.condition = np.random.randint(2)
        #self.condition = 1

    def on_close(self):
        """
        The WebSocket is closed.  Logger results need to be saved.
        """

        # Perform a Game Over in order to close out, e.g., logger
        # if hasattr(self, 'gameState'):
        #     self.on_game_over()

        # Kill the agent thread
        #        self.agent.stop()
        #        self.agent.kill()
        self.gameHandlerCallback.stop()
        # Get rid of the instance of builder and game
        # del self.logger

    #        del self.agent


    def randomArrage(self):
        reOrder = [3,2,4,5,1,8,9,10,7,6]
        random.shuffle(reOrder)
        print(reOrder)
        locList=[]
        for j in range(1,11):
            locList.append(self.tokens[j]['loc'])
        for i in range(1,11):
            self.tokens[i]['loc'] = locList[reOrder[i-1]-1]



    def load(self, args, path):
        # d = torch.load(path)
        # policy_net.load_state_dict(d['policy_net'])

        if(self.condition):
            args.seed = 0
            args.exp_name = 'one_hot81'
            self.tokens = one_hot_dict[args.seed]
            self.randomArrage()
            print(self.tokens)
        else:
            args.seed = 2
            args.exp_name = 'proto_fixed1'
            self.tokens = token_dict[args.seed]
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
        self.policy_net.load_state_dict(d['policy_net'], strict=False)

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
            if self.condition:
                args.hid_size = 81  # Was 128 for proto; 81 for one-hot
                args.num_proto = 81
                args.comm_dim = 81
                args.use_proto = False
                args.discrete_comm = True

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
            if not isinstance(args.num_actions, (list, tuple)):  # single action case
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
                args.seed = np.random.randint(0, 10000)
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
            reset_args = getfullargspec(self.env.reset).args
            if 'epoch' in reset_args:
                self.state = self.env.reset(epoch)
            else:
                self.state = self.env.reset()
            should_display = False

            # if should_display:
            #    self.env.display()
            self.stat = dict()
            self.info = dict()
            switch_t = -1

            self.prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

            # Process control and render initialization
            if self.currentTrial <= self.numTrial / 2:
                self.currentSession = self.firstSession
            else:
                self.currentSession = self.secondSession
            print(self.currentTrial, self.currentSession)
            if self.currentTrial > self.numTrial:
                # self.on_game_over()
                # self.currentSession = 'survey'
                self.write_message(self.gameStateJson)
                # self.on_close()

            self.done = False
            self.step = 0
            self.history = []
            self.moveRT = []
            self.selectedToken = None

            if self.currentSession == 'parent':
                self.done = True
                while self.done:
                    if 'epoch' in reset_args:
                        self.state = self.env.reset(epoch)
                    else:
                        self.state = self.env.reset()
                    for k in range(2):
                        should_display = False
                        t = self.t
                        misc = dict()
                        if t == 0 and self.args.hard_attn and self.args.commnet:
                            self.info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

                        self.info['record_comms'] = 1
                        # recurrence over time
                        if self.args.recurrent:
                            if self.args.rnn_type == 'LSTM' and t == 0:
                                self.prev_hid = self.policy_net.init_hidden(batch_size=self.state.shape[0])

                            x = [self.state, self.prev_hid]
                            action_out, value, self.prev_hid, filtered_comms = self.policy_net(x, self.info)

                            if (t + 1) % self.args.detach_gap == 0:
                                if self.args.rnn_type == 'LSTM':
                                    self.prev_hid = (self.prev_hid[0].detach(), self.prev_hid[1].detach())
                                else:
                                    self.prev_hid = self.prev_hid.detach()
                        else:
                            x = self.state
                            action_out, value, filtered_comms = self.policy_net(x, self.info)

                        # print(action_out)
                        print('filtered_comms', filtered_comms)
                        for i in self.tokens.keys():
                            if torch.allclose(torch.tensor(self.tokens[i]['raw']), filtered_comms, atol=1e-04):
                                self.selectedToken = i

                        print(self.selectedToken)

                        action = select_action(self.args, action_out)
                        # print(action)
                        action, actual = translate_action(self.args, self.env, action)
                        # actual[0] = self.humanAction
                        next_state, reward, done, info = self.env.step(actual)
                        # print(next_state)
                        # print(self.env.get_pp_loc_wrapper())

                        # store comm_action in info for next step
                        if self.args.hard_attn and self.args.commnet:
                            info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents,
                                                                                                           dtype=int)

                            # print("before ", stat.get('comm_action', 0), info['comm_action'][:self.args.nfriendly])
                            self.stat['comm_action'] = self.stat.get('comm_action', 0) + info['comm_action'][
                                                                                         :self.args.nfriendly]
                            self.all_comms.append(info['comm_action'][:self.args.nfriendly])
                            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                                self.stat['enemy_comm'] = self.stat.get('enemy_comm', 0) + info['comm_action'][
                                                                                           self.args.nfriendly:]

                        if 'alive_mask' in info:
                            misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
                        else:
                            misc['alive_mask'] = np.ones_like(reward)

                        # env should handle this make sure that reward for dead agents is not counted
                        # reward = reward * misc['alive_mask']

                        self.stat['reward'] = self.stat.get('reward', 0) + reward[:self.args.nfriendly]
                        if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                            self.stat['enemy_reward'] = self.stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

                        done = done or t >= self.args.max_steps - 1 or bool(self.env.get_reached_wrapper())

                        episode_mask = np.ones(reward.shape)
                        episode_mini_mask = np.ones(reward.shape)

                        if done:
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
                        self.done = done




                predator_loc, prey_loc = self.env.get_loc_wrapper()
                self.history.append({'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])})

                self.gameState = {
                    'players': {
                        'child': {'x': int(prey_loc[0, 1]), 'y': int(prey_loc[0, 0])},
                        'parent': {'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])},
                    },
                    'comm': self.tokens,
                    'selectedToken': self.selectedToken,
                    'step': self.step,
                    'best': self.best,
                    'done': False,
                    'currentTrial': self.currentTrial,
                    'humanRole': 'parent',
                    'history':self.history,
                    'condition':self.condition
                }
                # visibility should be handel by the environment, not here
                if abs(int(prey_loc[0, 1]) - int(predator_loc[0, 1])) > 1 or abs(
                    int(prey_loc[0, 0]) - int(predator_loc[0, 0])) > 1:
                    self.gameState['players'] = {
                        'parent': {'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])}
                    }
            elif self.currentSession == 'child':
                predator_loc, prey_loc = self.env.get_loc_wrapper()
                self.history.append({'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])})
                self.gameState = {
                    'players': {
                        'child': {'x': int(prey_loc[0, 1]), 'y': int(prey_loc[0, 0])},
                        'parent': {'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])},
                    },
                    'comm': self.tokens,
                    'selectedToken': self.selectedToken,
                    'step': self.step,
                    'best': self.best,
                    'done': False,
                    'currentTrial': self.currentTrial,
                    'humanRole': 'child',
                    'history': self.history,
                    'condition': self.condition
                }
                if self.currentTrial == 12 or self.currentTrial == 28:
                    self.gameState = {
                        'players': {
                        },
                        'attentionCheck':False,
                        'comm': self.tokens,
                        'selectedToken': None,
                        'step': self.step,
                        'best': self.best,
                        'done': False,
                        'currentTrial': self.currentTrial,
                        'humanRole': 'child',
                        'history': self.history,
                        'condition': self.condition
                    }


            self.gameStateJson = json.dumps(self.gameState)
            self.startTimer = time.time()
            self.write_message(self.gameStateJson)
            return

        if message_json["type"] == "command":

            # Pull out the command
            command = message_json["message"]

            # Pull out human role and check if function is correctly triggered
            humanRole = message_json["humanRole"]


            if self.currentSession != 'parent' or self.step >= self.args.max_steps:
                print('reject invalid action')
                return

            if command["command"] == "up":
                self.humanAction = 0
            elif command["command"] == "right":
                self.humanAction = 1
            elif command["command"] == "down":
                self.humanAction = 2
            elif command["command"] == "left":
                self.humanAction = 3
            else:
                return

            if self.step == 0:
                self.moveRT.append(time.time() - self.startTimer)
                self.lastMove = time.time()
            else:
                self.moveRT.append(time.time() - self.lastMove)
                self.lastMove = time.time()
            self.info['replace_comm'] = False
            self.step += 1







            next_state, reward, done, info = self.env.step([self.humanAction])

            self.done = done or self.step >= self.args.max_steps or bool(self.env.get_reached_wrapper())
            self.complete = bool(self.env.get_reached_wrapper())
            predator_loc, prey_loc = self.env.get_loc_wrapper()
            print(predator_loc, prey_loc)
            self.history.append({'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])})

            self.gameState = {
                'players': {
                    'child': {'x': int(prey_loc[0, 1]), 'y': int(prey_loc[0, 0])},
                    'parent': {'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])},
                },
                'comm': self.tokens,
                'selectedToken': self.selectedToken,
                'commRT': None,
                'moveRT': self.moveRT,
                'step': self.step,
                'best': self.best,
                'done': self.done,
                'complete': self.complete,
                'currentTrial': self.currentTrial,
                'humanRole': 'parent',
                'history': self.history,
                'condition':self.condition
            }




            if self.done:
                self.currentTrial += 1
                if self.step < self.best:
                    self.best = self.step
                self.save_log()

            # visibility should be handel by the environment, not here
            if not self.done:
                if abs(int(prey_loc[0, 1]) - int(predator_loc[0, 1])) > 1 or abs(
                    int(prey_loc[0, 0]) - int(predator_loc[0, 0])) > 1:
                    self.gameState['players'] = {
                        'parent': {'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])}
                    }
            self.gameStateJson = json.dumps(self.gameState)

            self.write_message(self.gameStateJson)


        if message_json["type"] == "comm":

            # Pull out human role and check if function is correctly triggered
            humanRole = message_json["humanRole"]

            if self.currentSession != 'child':
                return
            self.commTimer = time.time()
            self.commRT = self.commTimer - self.startTimer
            if self.currentTrial == 12 or self.currentTrial == 28:
                self.gameState = {
                    'players': {
                    },
                    'attentionCheck': True,
                    'comm': self.tokens,
                    'selectedToken': message_json['message'],
                    'commRT': self.commRT,
                    'moveRT': None,
                    'step': self.step,
                    'best': self.best,
                    'done': False,
                    'complete': False,
                    'currentTrial': self.currentTrial,
                    'humanRole': 'child',
                    'history': self.history,
                    'condition':self.condition
                }
                if message_json['message'] == 8: self.gameState['complete'] = True
                self.currentTrial += 1
                self.gameStateJson = json.dumps(self.gameState)
                self.write_message(self.gameStateJson)
                self.save_log()
                return
            self.currentTrial += 1
            while not self.done:
                self.step += 1
                should_display = False
                t = self.t
                misc = dict()
                if t == 0 and self.args.hard_attn and self.args.commnet:
                    self.info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

                # Hardcoded to record communication for agent 1 (prey)
                # UNCOMMENT FOR PROTOS
                # self.info['record_comms'] = 0
                # either 0 or 1 depending on which agent to inject comm vector for

                self.info['agent_id_replace'] = 1
                self.info['child_comm'] = torch.Tensor(self.tokens[message_json['message']]['raw'])
                # TEST COMM VEC COMMENT THE test_vec OUT WHEN USING ACTUAL MESSAGE
                # test_vec = [0.6093685361352408, 0.48378074925892295, 0.9301173165918591, 0.10678958236225718,
                #             0.23254922156143712, 0.1226728540104421, 0.8425635664122217, 0.68843780288164,
                #             0.10518988427018998]
                # self.info['child_comm'] = torch.Tensor(test_vec)
                self.info['replace_comm'] = True

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

                # print(action_out)

                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                next_state, reward, done, info = self.env.step(actual)
                # print(next_state)
                # print(self.env.get_pp_loc_wrapper())
                predator_loc, prey_loc = self.env.get_loc_wrapper()
                self.history.append({'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])})
                # print(self.prev_hid)
                print(reward)

                # store comm_action in info for next step
                if self.args.hard_attn and self.args.commnet:
                    info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents,
                                                                                                   dtype=int)

                    # print("before ", stat.get('comm_action', 0), info['comm_action'][:self.args.nfriendly])
                    self.stat['comm_action'] = self.stat.get('comm_action', 0) + info['comm_action'][
                                                                                 :self.args.nfriendly]
                    self.all_comms.append(info['comm_action'][:self.args.nfriendly])
                    if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                        self.stat['enemy_comm'] = self.stat.get('enemy_comm', 0) + info['comm_action'][
                                                                                   self.args.nfriendly:]

                if 'alive_mask' in info:
                    misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
                else:
                    misc['alive_mask'] = np.ones_like(reward)

                # env should handle this make sure that reward for dead agents is not counted
                # reward = reward * misc['alive_mask']

                self.stat['reward'] = self.stat.get('reward', 0) + reward[:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    self.stat['enemy_reward'] = self.stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

                done = done or t == self.args.max_steps - 1 or bool(self.env.get_reached_wrapper())

                episode_mask = np.ones(reward.shape)
                episode_mini_mask = np.ones(reward.shape)

                if done:
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
                self.done = done
                self.complete = bool(self.env.get_reached_wrapper())

                if self.done:
                    if self.step < self.best:
                        self.best = self.step
            predator_loc, prey_loc = self.env.get_loc_wrapper()
            self.gameState = {
                'players': {
                    'child': {'x': int(prey_loc[0, 1]), 'y': int(prey_loc[0, 0])},
                    'parent': {'x': int(predator_loc[0, 1]), 'y': int(predator_loc[0, 0])},
                },
                'comm': self.tokens,
                'selectedToken': message_json['message'],
                'commRT': self.commRT,
                'moveRT': None,
                'step': self.step,
                'best': self.best,
                'done': self.done,
                'complete':self.complete,
                'currentTrial': self.currentTrial,
                'humanRole': 'child',
                'history': self.history,
                'condition': self.condition
            }
            self.gameStateJson = json.dumps(self.gameState)

            self.write_message(self.gameStateJson)
            self.save_log()

        if message_json['type']=='survey':

            self.surveyResults = {
                'name':self.username,
                'condition':self.condition,
                "randomCode":message_json["randomCode"],
                "helpful":message_json["helpful"],
                "understand":message_json["understand"],
                "satisfy":message_json["satisfy"],
                "Post_difficulty": message_json["Post_difficulty"],
                "Post_how":message_json["Post_how"],
                "Post_feedback": message_json["Post_feedback"],
                "tokenLocation": message_json["tokenLocation"]
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
            if self.condition:
                group = 'onehot'
            else:
                group = 'proto'
            random_string = now + '_' + '_'.join([group,self.username])
            log_filename = 'survey_results_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)

            log_file_path = os.path.join(log_path, log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)

            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            with open(log_file_path, 'w') as outfile:
                json.dump(self.surveyResults, outfile)

        if self.gameState is not None:
            log_path = os.path.join(LOG_ROOT, self.username)

            # Create the folder, if it doesn't exist
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                print("Created path: ", log_path)

            # Create a unique filename
            now = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
            if self.condition:
                group = 'onehot'
            else:
                group = 'proto'
            random_string = now + '_' + '_'.join([group, self.username, str(self.currentTrial)])
            log_filename = 'game_log_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)


            log_file_path = os.path.join(log_path, log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)

            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            with open(log_file_path, 'w') as outfile:
                json.dump(self.gameState, outfile)

            # with open(meta_file_path,'w') as meta_file:
            #     meta_file.write('Agent Class: %s\n' % self.agent_class_name)


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
            if self.condition:
                group = 'onehot'
            else:
                group = 'proto'
            random_string = now + '_' + '_'.join([group,self.username,str(self.currentTrial)])
            log_filename = 'game_log_%s.json' % random_string
            # metadata_filename = 'game_log_%s.meta' % random_string
            existing_files = os.listdir(log_path)

            log_file_path = os.path.join(log_path, log_filename)
            # meta_file_path = os.path.join(log_path,metadata_filename)

            print("Log filename: %s" % log_file_path)
            # print("Metadata path: %s" % meta_file_path)

            with open(log_file_path, 'w') as outfile:
                json.dump(self.gameState, outfile)

            # with open(meta_file_path,'w') as meta_file:
            #     meta_file.write('Agent Class: %s\n' % self.agent_class_name)





if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = TSFWebApplication()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
