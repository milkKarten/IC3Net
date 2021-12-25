function real_game_time(game_time){
  var set_time = 180;//one trail time
  var time_n = set_time - game_time;
  if(time_n < 0){
    time_n =0;
    settimeout_win();
  }
  
  var m=parseInt(time_n/60);
  var s=parseInt(time_n%60);

  m = checkTime(m);
  s = checkTime(s);
  document.getElementById('timer').innerHTML =
  m + ":" + s;
  if(time_n == 0){
        clearInterval(time);
  }

  function checkTime(i) {
    if (i < 10) {i = "0" + i};
    return i;
  }

  function settimeout_win(){
    OpenWindow=window.open("", "newwin", "height=250, width=250,toolbar=no,scrollbars="+scroll+",menubar=no");
    OpenWindow.document.write("Time Out. Please click start button to continue your next trail.");
  }

}

function getParams(name, href) {
    var href = href || window.location.href,
      value = '';

    if (name) {
      var reg = new RegExp(name + '=([^&]*)', 'g');
      href.replace(reg, function($0, $1) {
        value = decodeURI($1);
      });
    } else {
      value = {};
      var reg = /\b(\w+)=([^\/&]*)/g;
      href.replace(reg, function($0, $1, $2) {
        value[$1] = decodeURI($2);
      });
    }
    return value;
};


function openwin() 
{ 
OpenWindow=window.open("", "newwin", "height=640, width=400,toolbar=no,scrollbars="+scroll+",menubar=no");
OpenWindow.document.write("<TITLE>Full Instruction</TITLE>") 
OpenWindow.document.write("<BODY BGCOLOR=#ffffff>") 
OpenWindow.document.write("<h1>Game Rule</h1>") 
OpenWindow.document.write("Coop-Space Fortress is a 2-D cooperative game where two players control spaceships to destroy a fortress. The fortress is located in the center of the screen and the other two ships around it are controlled by players. The first ship entering the hexagon area will be locked and shot by fortress. The Fortress becomes vulnerable when it is firing. Players die whenever they hit any obstacles (e.g. boundaries, missiles, the fortress). You will gain 100 scores every time your team successes in destroying the fortress, lose 100 scores every time one of the player dies, and lose 20 scores every time your team miss hit the shield. The game resets every time after either fortress or both players are killed.") 
OpenWindow.document.write("<h1>Role Assignment</h1>") 
OpenWindow.document.write("A common strategy of this game is assigning two players roles of either bait or shooter. The bait  tries to attract the fortress' attention by entering the inner hexagon where it is vulnerable to the fortress. When the fortress attempts to shoot at the bait, it's shield lifts making it vulnerable. The other player in the role of shooter can now shoot at the fortress and destroy it. The team performance is measured by the number of fortress players kill.")
OpenWindow.document.write("<h1>Procedure</h1>") 
OpenWindow.document.write("You will be assigned a role of either Bait or Shooter, which will remain the same during the whole experiment session. You will play the game teaming up with different AI partners in the complementary role. Your task is to collaborate with your partner to kill the fortress as much as you can in each 1-min trial. The strategy of your partner may vary across trials so you need to adapt to it for a better team performance.")
OpenWindow.document.write("<h1>Training Process</h1>") 
OpenWindow.document.write("Here is a single-player training session for you to get familiar with this game and the responsibility of your designated role. Once you reach the minimum performance requirement, you can pass the training and process the experiment.")
OpenWindow.document.write("<h1>Control</h1>") 
OpenWindow.document.write("D or right arrow - clockwise rotation , A or left arrow - counterclockwise rotation, J of F - shoot, W or up arrow - thrust.")
OpenWindow.document.write("</BODY>") 
OpenWindow.document.write("</HTML>") 
OpenWindow.document.close() 
} 


const ws = new WebSocket("ws://"+location.host+"/tsfsocket");

ws.onopen = function() 
{
    console.log('WebSocket Client Connected');
    ws.send('Hi this is web client.');
};

ws.onmessage = function(e) 
{
    //console.log("Received: '" + e.data + "'");
    var jsonObject = JSON.parse(e.data);
    draw(jsonObject);
};


function info_send(name){
  var message = new Object();
  message.type = "information";
  message.message = "start";
  message.name = name;
  ws.send(JSON.stringify(message));
}


function draw(frame_info){
    var canvas = document.getElementById("canvas");
    canvas.style.backgroundColor='black'
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
	   
    ctx.translate(355, 310);
    ctx.scale(1, -1);

    $("#score").val(frame_info.score);
    real_game_time(frame_info.time);
 
    //draw player
    for (var key in frame_info.players) {
        if (frame_info.players.hasOwnProperty(key)) {
            trans_rota(ctx,frame_info.players[key].position.x,frame_info.players[key].position.y,frame_info.players[key].angle);
            if(key == 1 & frame_info.players[key].alive == true){
                player1Shape(ctx);
            }
            else if(key == 0 & frame_info.players[key].alive == true){
                player0Shape(ctx);
            }             
            rever_trans_rota(ctx, frame_info.players[key].position.x,frame_info.players[key].position.y,frame_info.players[key].angle);
        }
    }
     
    //draw fortress
    for (var key in frame_info.fortresses) {
        if (frame_info.fortresses.hasOwnProperty(key)) {
            ctx.strokeStyle = "rgb(0,70,0)";
            plotHexagon(ctx, frame_info.fortresses[key].shield.position.x, frame_info.fortresses[key].shield.position.y, frame_info.fortresses[key].activationRegion.radius, frame_info.fortresses[key].activationRegion.angle, false);
              

            if(frame_info.fortresses[key].alive == true){
                ctx.strokeStyle = "green";

                plotHexagon(ctx, frame_info.fortresses[key].shield.position.x, frame_info.fortresses[key].shield.position.y, frame_info.fortresses[key].shield.radius, frame_info.fortresses[key].shield.angle, frame_info.fortresses[key].shield.vulnerable);
                trans_rota(ctx,frame_info.fortresses[key].x, frame_info.fortresses[key].y, frame_info.fortresses[key].shield.angle);
                fortressShape(ctx);
                rever_trans_rota(ctx, frame_info.fortresses[key].x, frame_info.fortresses[key].y, frame_info.fortresses[key].shield.angle);
            }             
        }
    }
     
    //draw missiles
    for (var key in frame_info.missiles) {
        if (frame_info.missiles.hasOwnProperty(key)) {
            trans_rota(ctx,frame_info.missiles[key].position.x, frame_info.missiles[key].position.y, frame_info.missiles[key].angle);
            missileShape(ctx);
            rever_trans_rota(ctx, frame_info.missiles[key].position.x, frame_info.missiles[key].position.y, frame_info.missiles[key].angle);
        }
    }
          
    //draw shells
    for (var key in frame_info.shells) {
        if (frame_info.shells.hasOwnProperty(key)) {
            trans_rota(ctx,frame_info.shells[key].position.x, frame_info.shells[key].position.y, frame_info.shells[key].angle);
            shellShape(ctx);
            rever_trans_rota(ctx,frame_info.shells[key].position.x, frame_info.shells[key].position.y, frame_info.shells[key].angle);
        }
    }
}


function plotHexagon(ctx, x, y, radius, angle, flag){
    var c = Math.cos(angle * Math.PI / 180);
    var s = Math.sin(angle * Math.PI / 180);

    var points=[];
    points[0] = [x + c*Math.floor(-radius), y + s*Math.floor(-radius)];
    points[1] = [x + c*Math.floor(-0.5*radius) - s*Math.floor(-radius*Math.sin(2*Math.PI/3)), y + s*Math.floor(-0.5*radius) + c*Math.floor(-radius*Math.sin(2*Math.PI/3))];
    points[2] = [x + c*Math.floor(0.5*radius) - s*Math.floor(-radius*Math.sin(2*Math.PI/3)), y + s*Math.floor(0.5*radius) + c*Math.floor(-radius*Math.sin(2*Math.PI/3))];
    points[3] = [x + c*Math.floor(radius), y + s*Math.floor(radius)];
    points[4] = [x + c*Math.floor(0.5*radius) - s*Math.floor(radius*Math.sin(2*Math.PI/3)), y + s*Math.floor(0.5*radius) + c*Math.floor(radius*Math.sin(2*Math.PI/3))];
    points[5] = [x + c*Math.floor(-0.5*radius) - s*Math.floor(radius*Math.sin(2*Math.PI/3)), y + s*Math.floor(-0.5*radius) + c*Math.floor(radius*Math.sin(2*Math.PI/3))];

    if (flag == false){
        ctx.beginPath();
        for (var i = 0; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1])
        }
        ctx.lineTo(points[0][0], points[0][1])
        ctx.stroke();
    }
    else{
        ctx.beginPath();
        for (var i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1])
        }
        ctx.stroke();
    }
  
}

function player0Shape(ctx){
    ctx.beginPath();
    ctx.strokeStyle = "#ff00ff";
    ctx.lineTo(-18,0);
    ctx.lineTo(18,0);
    ctx.moveTo(-18,18);
    ctx.lineTo(0,0);
    ctx.moveTo(0,0);
    ctx.lineTo(-18,-18);
    ctx.stroke();
}

function player1Shape(ctx){
    ctx.beginPath();
    ctx.strokeStyle = "Aqua";
    ctx.lineTo(-18,0);
    ctx.lineTo(18,0);
    ctx.moveTo(-18,18);
    ctx.lineTo(0,0);
    ctx.moveTo(0,0);
    ctx.lineTo(-18,-18);
    ctx.stroke();
}

function fortressShape(ctx){
    ctx.beginPath();
    ctx.strokeStyle = "yellow";
    ctx.lineTo(0,0);
    ctx.lineTo(36,0);
    ctx.moveTo(0,-18);
    ctx.lineTo(18,-18);
    ctx.moveTo(18,-18);
    ctx.lineTo(18,18);
    ctx.moveTo(18,18);
    ctx.lineTo(0,18);
    ctx.stroke();
}

function missileShape(ctx){
    ctx.beginPath();
    ctx.strokeStyle = "white";
    ctx.lineTo(0,0);
    ctx.lineTo(-25,0);
    ctx.moveTo(0,0);
    ctx.lineTo(-5,5);
    ctx.moveTo(0,0);
    ctx.lineTo(-5,-5);
    ctx.stroke();
}

function shellShape(ctx){
    ctx.beginPath();
    ctx.strokeStyle = "red";
    ctx.lineTo(-8,0);
    ctx.lineTo(0,-6);
    ctx.lineTo(16,0);
    ctx.lineTo(0,6);
    ctx.lineTo(-8,0);
    ctx.stroke();
}

function trans_rota(ctx, x, y, angle){
    ctx.translate(x,y)
    ctx.rotate(angle * Math.PI / 180)
}

function rever_trans_rota(ctx, x, y, angle){
    ctx.rotate(-angle * Math.PI / 180)
    ctx.translate(-x,-y)
}
 
