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
OpenWindow.document.write("You will play the role of either a parent or a child and complete the task with an AI partner in the complementary role. The parent need to find the lost child, and she can only see the child in a close range. The goal of the child is to effectively communicate its location so that the parent can travel to the lost child.")
OpenWindow.document.write("<h1>Procedure</h1>")
OpenWindow.document.write("You will need to complete 20 trials as a parent and another 20 trials as a child. A trial begins with the parent and child spawned at random locations, and ends when the parent reaches to the same location with the child or exceeds maximum steps allowed.")
OpenWindow.document.write("<h1>Communication Task</h1>")
OpenWindow.document.write("In the communication panel, there are several tokens for the child to communicate its location to the parent. Each communication token refers to a specific location in the task environment. The child should select a token to inform the parent her location, and the parent should search for the child according to received token. Your task is to learn to use those tokens to collaborate with your AI partner in both parent and child roles. Your AI partner is trained to have a good understanding of those tokens, so you should be able to rely on its behavior to learn the communication. Your compensation is based on <b>task performance</b> (e.g. number of completed trials and steps taken in each trials).")
OpenWindow.document.write("<h1>Control</h1>")
OpenWindow.document.write("Parent control: D or right arrow - go right, A or left arrow - go left, W or up arrow - go up, S or down arrow - go down. Child control: select communication contents using buttons on the right side")
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
    if (jsonObject.currentTrial>40){survey()}
    else{
        if (jsonObject.humanRole=='parent'){
            hideButtons();
            draw(jsonObject);
        }
        else{
            displayButtons();
            if (jsonObject.done==false){draw(jsonObject);}
            else{drawC(jsonObject);}

        }
    }

};

//state = {
//    'players':{
//        'child':{'x':3,'y':3},
//        'parent':{'x':1,'y':1},
//    },
//    'comm':{
//        'token1':{'x':4,'y':4,'index':1},
//        'token2':{'x':1,'y':2,'index':2},
//        'token3':{'x':4,'y':0,'index':3},
//        'token4':{'x':0,'y':4,'index':4},
//        'token5':{'x':0,'y':0,'index':5}
//    },
//    'selectedToken':1,
//    'step':0,
//    'best':999,
//    'humanRole':'parent'
//}
//
//draw(state)

function survey(){
    console.log('enter survey')
    document.getElementById('survey').style.visibility = 'visible';
    document.removeEventListener('keydown',keyDownCheck,false);
    hideButtons();

    var Token = function(x, y, index) {

      this.x = x;
      this.y = y;
      this.index = index;
      this.width = 24;
      this.height = 24;
      this.isDragging = false;

      this.render = function(ctx) {


        ctx.save();

        ctx.beginPath();
        ctx.rect(this.x - this.width * 0.5, this.y - this.height * 0.5, this.width, this.height);
        ctx.fillStyle = 'gray';
        ctx.fill();
        ctx.font = '24px serif';
        ctx.strokeStyle = "black";
        ctx.strokeText(this.index.toString(),this.x- this.width * 0.25,this.y + this.height * 0.5);

        ctx.restore();


      }
    }


    var MouseTouchTracker = function(canvas, callback){

      function processEvent(evt) {
        var rect = canvas.getBoundingClientRect();
        var offsetTop = rect.top;
        var offsetLeft = rect.left;

        if (evt.touches) {
          return {
            x: evt.touches[0].clientX - offsetLeft,
            y: evt.touches[0].clientY - offsetTop
          }
        } else {
          return {
            x: evt.clientX - offsetLeft,
            y: evt.clientY - offsetTop
          }
        }
      }

      function onDown(evt) {
        evt.preventDefault();
        var coords = processEvent(evt);
        callback('down', coords.x, coords.y);
      }

      function onUp(evt) {
        evt.preventDefault();
        callback('up');
      }

      function onMove(evt) {
        evt.preventDefault();
        var coords = processEvent(evt);
        callback('move', coords.x, coords.y);
      }

      canvas.ontouchmove = onMove;
      canvas.onmousemove = onMove;

      canvas.ontouchstart = onDown;
      canvas.onmousedown = onDown;
      canvas.ontouchend = onUp;
      canvas.onmouseup = onUp;
    }

    function isHit(shape, x, y) {

        if (x > shape.x - shape.width * 0.5 && y > shape.y - shape.height * 0.5 && x < shape.x + shape.width - shape.width * 0.5 && y < shape.y + shape.height - shape.height * 0.5) {
            return true;
        }

        return false;
    }

    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    var startX = 0;
    var startY = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBoard(ctx);
    document.getElementById('sessionReminder').innerHTML = 'Please drag tokens to corresponding locations you think they refer to, and fill the following survey.'
    var token1 = new Token(25, 25, 1);
    var token2 = new Token(50, 50, 2);
    var token3 = new Token(100, 100, 3);
    var token4 = new Token(150, 150, 4);
    var token5 = new Token(200, 200, 5);
    var token6 = new Token(250, 250, 6);
    var token7 = new Token(300, 300, 7);
    var token8 = new Token(350, 350, 8);
    var token9 = new Token(400, 400, 9);
    var token10 = new Token(450, 450, 10);


    token1.render(ctx);
    token2.render(ctx);
    token3.render(ctx);
    token4.render(ctx);
    token5.render(ctx);
    token6.render(ctx);
    token7.render(ctx);
    token8.render(ctx);
    token9.render(ctx);
    token10.render(ctx);

    var mtt = new MouseTouchTracker(canvas,
      function(evtType, x, y) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        switch(evtType) {

          case 'down':
            startX = x;
            startY = y;

            if (isHit(token1, x, y)) {
              token1.isDragging = true;
            }
            if (isHit(token2, x, y)) {
              token2.isDragging = true;
            }
            if (isHit(token3, x, y)) {
              token3.isDragging = true;
            }
            if (isHit(token4, x, y)) {
              token4.isDragging = true;
            }
            if (isHit(token5, x, y)) {
              token5.isDragging = true;
            }
            if (isHit(token6, x, y)) {
              token6.isDragging = true;
            }
            if (isHit(token7, x, y)) {
              token7.isDragging = true;
            }
            if (isHit(token8, x, y)) {
              token8.isDragging = true;
            }
            if (isHit(token9, x, y)) {
              token9.isDragging = true;
            }
            if (isHit(token10, x, y)) {
              token10.isDragging = true;
            }


            break;

          case 'up':

            token1.isDragging = false;
            token2.isDragging = false;
            token3.isDragging = false;
            token4.isDragging = false;
            token5.isDragging = false;
            token6.isDragging = false;
            token7.isDragging = false;
            token8.isDragging = false;
            token9.isDragging = false;
            token10.isDragging = false;


            break;

          case 'move':
            var dx = x - startX;
            var dy = y - startY;
            startX = x;
            startY = y;

            if (token1.isDragging) {
              token1.x += dx;
              token1.y += dy;
            }
            if (token2.isDragging) {
              token2.x += dx;
              token2.y += dy;
            }
            if (token3.isDragging) {
              token3.x += dx;
              token3.y += dy;
            }
            if (token4.isDragging) {
              token4.x += dx;
              token4.y += dy;
            }
            if (token5.isDragging) {
              token5.x += dx;
              token5.y += dy;
            }
            if (token6.isDragging) {
              token6.x += dx;
              token6.y += dy;
            }
            if (token7.isDragging) {
              token7.x += dx;
              token7.y += dy;
            }
            if (token8.isDragging) {
              token8.x += dx;
              token8.y += dy;
            }
            if (token9.isDragging) {
              token9.x += dx;
              token9.y += dy;
            }
            if (token10.isDragging) {
              token10.x += dx;
              token10.y += dy;
            }


            break;
        }
        drawBoard(ctx);
        token1.render(ctx);
        token2.render(ctx);
        token3.render(ctx);
        token4.render(ctx);
        token5.render(ctx);
        token6.render(ctx);
        token7.render(ctx);
        token8.render(ctx);
        token9.render(ctx);
        token10.render(ctx);



      }
    );

}

function attachTokenResults(){
    var tokenLocation = {
        1: {x:token1.x, y:token1.y},
        2: {x:token2.x, y:token2.y},
        3: {x:token3.x, y:token3.y},
        4: {x:token4.x, y:token4.y},
        5: {x:token5.x, y:token5.y},
        6: {x:token6.x, y:token6.y},
        7: {x:token7.x, y:token7.y},
        8: {x:token8.x, y:token8.y},
        9: {x:token9.x, y:token9.y},
        10: {x:token10.x, y:token10.y},
    }
    document.getElementById("tokenLocation").value = tokenLocation;
}

function hideButtons(){
    document.getElementById('token1').style.visibility = 'hidden';
    document.getElementById('token2').style.visibility = 'hidden';
    document.getElementById('token3').style.visibility = 'hidden';
    document.getElementById('token4').style.visibility = 'hidden';
    document.getElementById('token5').style.visibility = 'hidden';
    document.getElementById('token6').style.visibility = 'hidden';
    document.getElementById('token7').style.visibility = 'hidden';
    document.getElementById('token8').style.visibility = 'hidden';
    document.getElementById('token9').style.visibility = 'hidden';
    document.getElementById('token10').style.visibility = 'hidden';
    }

function displayButtons(){
    document.getElementById('token1').style.visibility = 'visible';
    document.getElementById('token2').style.visibility = 'visible';
    document.getElementById('token3').style.visibility = 'visible';
    document.getElementById('token4').style.visibility = 'visible';
    document.getElementById('token5').style.visibility = 'visible';
    document.getElementById('token6').style.visibility = 'visible';
    document.getElementById('token7').style.visibility = 'visible';
    document.getElementById('token8').style.visibility = 'visible';
    document.getElementById('token9').style.visibility = 'visible';
    document.getElementById('token10').style.visibility = 'visible';
    }

function info_send(name){
  var message = new Object();
  document.getElementById("startGame").style.visibility = 'hidden';
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

    if (frame_info.humanRole == 'child'){
        document.getElementById('sessionReminder').innerHTML = 'You are the child, select a token to communicate your location to the parent.'
    }
    if (frame_info.humanRole == 'parent'){
        document.getElementById('sessionReminder').innerHTML = 'You are the parent, use WASD to move and search for the child based on the communication your received.'
    }
    document.getElementById('best').innerHTML = frame_info.best.toString();
    document.getElementById('step').innerHTML = frame_info.step.toString();
    if (frame_info.hasOwnProperty('currentTrial')) {
        document.getElementById('trial').innerHTML = frame_info.currentTrial.toString();
    }
    //draw board
    drawBoard(ctx);
    //draw parent's vision

    //draw player
    if (frame_info.done){
        console.log('new_trial');
        drawResult(ctx,frame_info.players['child'].x,frame_info.players['child'].y,frame_info);
        setTimeout(() => {info_send();}, 2500);
    }
    else{
        drawVision(ctx,frame_info.players['parent'].x,frame_info.players['parent'].y);
        for (var key in frame_info.players) {
            if (frame_info.players.hasOwnProperty(key)) {
                if(key == 'child'){
                    drawChild(ctx,frame_info.players[key].x,frame_info.players[key].y,frame_info.humanRole);
                }
                else if(key == 'parent'){

                    drawParent(ctx,frame_info.players[key].x,frame_info.players[key].y,frame_info.humanRole);
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
            if (key == frame_info.selectedToken){drawToken(ctx,frame_info.comm[key].loc[0],frame_info.comm[key].loc[1],key.toString(),true);}
            else{drawToken(ctx,frame_info.comm[key].loc[0],frame_info.comm[key].loc[1],key.toString(),false);}
        }
    }
}




function drawC(frame_info){
    console.log('start drawC')
    hideButtons();
    if (frame_info.humanRole == 'child'){
        document.getElementById('sessionReminder').innerHTML = 'You are the child, select a token to communicate your location to the parent.'
    }
    if (frame_info.humanRole == 'parent'){
        document.getElementById('sessionReminder').innerHTML = 'You are the parent, use WASD to move and search for the child based on the communication your received.'
    }
    document.getElementById('best').innerHTML = frame_info.best.toString();

    if (frame_info.hasOwnProperty('currentTrial')) {
        document.getElementById('trial').innerHTML = frame_info.currentTrial.toString();
    }
    var canvas = document.getElementById("comm");
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
     //draw comm space
    drawSpace(ctx);
    //draw comm tokens
    for (var key in frame_info.comm){
        if (key == frame_info.selectedToken){drawToken(ctx,frame_info.comm[key].loc[0],frame_info.comm[key].loc[1],key.toString(),true);}
        else{drawToken(ctx,frame_info.comm[key].loc[0],frame_info.comm[key].loc[1],key.toString(),false);}
    }
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    for (var i = 0; i<=frame_info.step;i++){
        console.log('enter for loop');
        updateLoc(frame_info,i,ctx);
    }
    console.log('new_trial');
    setTimeout(() => {drawResult(ctx,frame_info.players['child'].x,frame_info.players['child'].y,frame_info);}, i*500+500);
    setTimeout(() => {displayButtons();info_send();}, i*500+2500);
}


function updateLoc(frame_info,i,ctx){
    setTimeout(function() {
        document.getElementById('step').innerHTML = i.toString();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        //draw board
        drawBoard(ctx);
        //draw parent's vision
        var parentLocation = frame_info.history[i];
        console.log(parentLocation);
        drawVision(ctx,parentLocation.x,parentLocation.y);
        //draw player
        for (var key in frame_info.players) {
            if (frame_info.players.hasOwnProperty(key)) {
                if(key == 'child'){
                    drawChild(ctx,frame_info.players[key].x,frame_info.players[key].y,frame_info.humanRole);
                }
                else if(key == 'parent'){
                    drawParent(ctx,parentLocation.x,parentLocation.y,frame_info.humanRole);
                }
            }
        }
    }, i*500);
}


function drawToken(ctx,x,y,text,selected){
    if(selected){ctx.strokeStyle = "red";}
    else{ctx.strokeStyle = "black";}
    ctx.font = '24px serif';
    ctx.strokeText(text,x*50+50,y*50+60);
}


function drawParent(ctx,x,y,humanRole){
    const image = document.getElementById('parent');
    ctx.drawImage(image, 38, 68, 27, 27, x*50+40, y*50+40, 50, 50);
    if(humanRole=='parent'){ctx.strokeStyle = "red";}
    else{ctx.strokeStyle = "black";}
    ctx.font = '16px serif';
    ctx.strokeText('Parent',x*50+40,y*50+40);
}

function drawChild(ctx,x,y,humanRole){
    const image = document.getElementById('child');
    ctx.drawImage(image, 38, 66, 27, 27, x*50+40, y*50+40, 50, 50);
    if(humanRole=='child'){ctx.strokeStyle = "red";}
    else{ctx.strokeStyle = "black";}
    ctx.font = '16px serif';
    ctx.strokeText('Child',x*50+40,y*50+40);

}

function drawResult(ctx,x,y,frame_info){
    if (frame_info.humanRole == 'child'){
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawBoard(ctx);
        if (frame_info.complete){
            console.log('child, succ')

            var minimum = Math.abs(frame_info.history[0].x-x)+Math.abs(frame_info.history[0].y-y);
            console.log(minimum);
            for (let i = 0; i<frame_info.history.length; i ++){
                drawParent(ctx,frame_info.history[i].x,frame_info.history[i].y);
            }
            const image = document.getElementById('heart');
            ctx.strokeStyle = "red";
            ctx.drawImage(image, x*50+50, y*50+50, 30, 30);
            ctx.font = '20px serif';
            ctx.strokeText('Actual steps taken: '+ frame_info.step.toString(),0,20);
            ctx.strokeText('Minimum steps possible: ' + minimum.toString(),250,20);
        }
        else{
            console.log('child, fail')
            const image = document.getElementById('failure');
            ctx.strokeStyle = "black";
            ctx.drawImage(image, x*50+50, y*50+50, 30, 30);
            var minimum = Math.abs(frame_info.history[0].x-x)+Math.abs(frame_info.history[0].y-y);
            for (let i = 0; i<frame_info.history.length; i ++){
                drawParent(ctx,frame_info.history[i].x,frame_info.history[i].y);
            }

            ctx.font = '20px serif';
            ctx.strokeText('Actual steps taken: '+ frame_info.step.toString(),0,20);
            ctx.strokeText('Minimum steps possible: ' + minimum.toString(),250,20);
        }
    }
    if (frame_info.humanRole == 'parent'){
        if (frame_info.complete){
            console.log('parent, succ')
            for (let i = 0; i<frame_info.history.length; i ++){
                drawParent(ctx,frame_info.history[i].x,frame_info.history[i].y);
            }
            const image = document.getElementById('heart');
            ctx.drawImage(image, x*50+50, y*50+50, 30, 30);
            ctx.strokeStyle = "red";
            text = 'Task completed with token'+ frame_info.selectedToken.toString()
            ctx.font = '20px serif';
            ctx.strokeText(text,120,20);
        }
        else{
            console.log('parent, fail')
            const image = document.getElementById('child');
            ctx.drawImage(image, 38, 66, 27, 27, x*50+40, y*50+40, 50, 50);
            for (let i = 0; i<frame_info.history.length; i ++){
                drawParent(ctx,frame_info.history[i].x,frame_info.history[i].y);
            }
            ctx.strokeStyle = "black";
            text = 'Task failed with token'+ frame_info.selectedToken.toString()
            ctx.font = '20px serif';
            ctx.strokeText(text,120,20);
        }
    }
}



function drawSpace(ctx){
    ctx.arc(250,250,225,0, 2 * Math.PI);
    ctx.strokeStyle = "black";
    ctx.stroke();
}

function drawBoard(ctx){
    // Box width
    var bw = 450;
    // Box height
    var bh = 450;
    // Padding
    var p = 40;
    for (var x = 0; x <= bw; x += 50) {
            ctx.moveTo(0.5 + x + p, p);
            ctx.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x <= bh; x += 50) {
            ctx.moveTo(p, 0.5 + x + p);
            ctx.lineTo(bw + p, 0.5 + x + p);
        }
        ctx.strokeStyle = "black";
        ctx.stroke();
}

function drawVision(ctx,x,y){
    // Box width
    var bw = 450;
    // Box height
    var bh = 450;
    // Padding
    var p = 40;
    for (var i = x-1; i <= x+1; i += 1) {
            for (var j = y-1; j <= y+1; j += 1) {
            if(i<0 || j<0){continue;}
            if(i>8 || j>8){continue;}
            ctx.fillStyle = 'gray';
            ctx.fillRect(i*50+p+0.5,j*50+p+0.5, 50-0.5, 50-0.5);
    }
    }

}

function commSend(token){
  var message = new Object();
  message.type = "comm";
  message.message = token;
  message.humanRole = 'child';
  hideButtons();
//  state = state_update(state,message);
//  draw(state);
  ws.send(JSON.stringify(message));
}

//function state_update(state,message){
//  if(message.type == 'command'){
//    if(message.message.command == "up" && state.players['parent'].y-1>=0){state.players['parent'].y-=1}
//    if(message.message.command == "left" && state.players['parent'].x-1>=0){state.players['parent'].x-=1}
//    if(message.message.command == "right" && state.players['parent'].x+1<=4){state.players['parent'].x+=1}
//    if(message.message.command == "down" && state.players['parent'].y+1<=4){state.players['parent'].y+=1}
//    state.step+=1;
//    return state
//  }
//  if(message.type == 'comm'){
//    for (var token in state.comm){
//        if (state.comm[token].selected){state.comm[token].selected = false}
//    }
//    state.comm[message.message].selected=true;
//    return state
//  }
//}

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
  message.humanRole = 'parent'
//  state = state_update(state,message);
//  draw(state);
  ws.send(JSON.stringify(message));
}


function keyDownCheck(event){
  var e= event||window.event||arguments.callee.caller.arguments[0];
  console.log(e.keyCode);
  message_send(e.keyCode, true);
}



