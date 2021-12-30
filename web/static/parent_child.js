//  function real_game_time(game_time){
//    var set_time = 180;//one trail time
//    var time_n = set_time - game_time;
//    if(time_n < 0){
//      time_n =0;
//      settimeout_win();
//  }
//
//  var m=parseInt(time_n/60);
//  var s=parseInt(time_n%60);
//
//  m = checkTime(m);
//  s = checkTime(s);
//  document.getElementById('timer').innerHTML =
//  m + ":" + s;
//  if(time_n == 0){
//        clearInterval(time);
//  }
//
//  function checkTime(i) {
//    if (i < 10) {i = "0" + i};
//    return i;
//  }
//
//  function settimeout_win(){
//    OpenWindow=window.open("", "newwin", "height=250, width=250,toolbar=no,scrollbars="+scroll+",menubar=no");
//    OpenWindow.document.write("Time Out. Please click start button to continue your next trail.");
//  }
//
//}

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


//const ws = new WebSocket("ws://"+location.host+"/tsfsocket");
//
//ws.onopen = function()
//{
//    console.log('WebSocket Client Connected');
//    ws.send('Hi this is web client.');
//};
//
//ws.onmessage = function(e)
//{
//    //console.log("Received: '" + e.data + "'");
//    var jsonObject = JSON.parse(e.data);
//    draw(jsonObject);
//};

state = {
    'players':{
        'child':{'x':3,'y':3},
        'parent':{'x':1,'y':1},
    },
    'comm':{
        'token1':{'x':4,'y':4,'selected':true},
        'token2':{'x':1,'y':2,'selected':false},
        'token3':{'x':4,'y':0,'selected':false},
        'token4':{'x':0,'y':4,'selected':false},
        'token5':{'x':0,'y':0,'selected':false}
    },
    'step':0,
    'best':7
}

draw(state)

function info_send(name){
  var message = new Object();
  message.type = "information";
  message.message = "start";
  message.name = name;
  ws.send(JSON.stringify(message));
}


function draw(frame_info){
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);


    document.getElementById('best').innerHTML = frame_info.best.toString();
    document.getElementById('step').innerHTML = frame_info.step.toString();
    //draw board
    drawBoard(ctx)
    //draw parent's vision
    drawVision(ctx,frame_info.players['parent'].x,frame_info.players['parent'].y);
    //draw player
    for (var key in frame_info.players) {
        if (frame_info.players.hasOwnProperty(key)) {
            if(key == 'child'){
                drawChild(ctx,frame_info.players[key].x,frame_info.players[key].y);
            }
            else if(key == 'parent'){
                drawParent(ctx,frame_info.players[key].x,frame_info.players[key].y);

            }
        }
    }
    var canvas = document.getElementById("comm");
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
     //draw comm space
    drawSpace(ctx)
    //draw comm tokens
    for (var key in frame_info.comm){
        if (frame_info.comm[key].selected){drawToken(ctx,frame_info.comm[key].x,frame_info.comm[key].y,key,true);}
        else{drawToken(ctx,frame_info.comm[key].x,frame_info.comm[key].y,key,false);}
    }


}


function drawToken(ctx,x,y,text,selected){
    if(selected){ctx.strokeStyle = "red";}
    else{ctx.strokeStyle = "black";}
    ctx.font = '24px serif';
    ctx.strokeText(text,x*80+50,y*80+60);
}


function drawParent(ctx,x,y){
    const image = document.getElementById('parent');
    ctx.drawImage(image, 38, 68, 27, 27, x*80+50, y*80+50, 75, 75);
}

function drawChild(ctx,x,y){
    const image = document.getElementById('child');
    ctx.drawImage(image, 38, 66, 27, 27, x*80+50, y*80+50, 75, 75);
}

function drawSpace(ctx){
    // Box width
    var bw = 400;
    // Box height
    var bh = 400;
    // Padding
    var p = 40;
    for (var x = 0; x <= bw; x += 400) {
            ctx.moveTo(0.5 + x + p, p);
            ctx.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x <= bh; x += 400) {
            ctx.moveTo(p, 0.5 + x + p);
            ctx.lineTo(bw + p, 0.5 + x + p);
        }
        ctx.strokeStyle = "black";
        ctx.stroke();
}

function drawBoard(ctx){
    // Box width
    var bw = 400;
    // Box height
    var bh = 400;
    // Padding
    var p = 40;
    for (var x = 0; x <= bw; x += 80) {
            ctx.moveTo(0.5 + x + p, p);
            ctx.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x <= bh; x += 80) {
            ctx.moveTo(p, 0.5 + x + p);
            ctx.lineTo(bw + p, 0.5 + x + p);
        }
        ctx.strokeStyle = "black";
        ctx.stroke();
}

function drawVision(ctx,x,y){
    // Box width
    var bw = 400;
    // Box height
    var bh = 400;
    // Padding
    var p = 40;
    for (var i = x-1; i <= x+1; i += 1) {
            for (var j = y-1; j <= y+1; j += 1) {
            if(i<0 || j<0){continue;}
            if(i>4 || j>4){continue;}
            ctx.fillStyle = 'gray';
            ctx.fillRect(i*80+p+0.5,j*80+p+0.5, 80-0.5, 80-0.5);
    }
    }

}

function commUpdate(token){
  var message = new Object();
  message.type = "comm"
  message.message = token;
  state = state_update(state,message);
  draw(state);
//  ws.send(JSON.stringify(message));
}

function state_update(state,message){
  if(message.type == 'command'){
    if(message.message.command == "up" && state.players['parent'].y-1>=0){state.players['parent'].y-=1}
    if(message.message.command == "left" && state.players['parent'].x-1>=0){state.players['parent'].x-=1}
    if(message.message.command == "right" && state.players['parent'].x+1<=4){state.players['parent'].x+=1}
    if(message.message.command == "down" && state.players['parent'].y+1<=4){state.players['parent'].y+=1}
    state.step+=1;
    return state
  }
  if(message.type == 'comm'){
    for (var token in state.comm){
        if (state.comm[token].selected){state.comm[token].selected = false}
    }
    state.comm[message.message].selected=true;
    return state
  }
}

function message_send(key, flag) {
  var command = new Object();
  command.player = 'parent';
  if (key == 87 || key == 38){
    command.command = "up" //speed up
  }
  else if (key == 65 || key == 37){
    command.command = "left"
  }
  else if (key == 68 || key == 39){
    command.command = "right"
  }
  else if (key == 83 || key == 40){
    command.command = "down"
  }
  else{
    command.command = null
  }
  command.isPress = flag;

  var message = new Object();
  message.type = "command"
  message.message = command;
  state = state_update(state,message);
  draw(state);
//  ws.send(JSON.stringify(message));
}


function keyDownCheck(event){
  var e= event||window.event||arguments.callee.caller.arguments[0];
  console.log(e.keyCode);
  message_send(e.keyCode, true);
}



