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
OpenWindow.document.write("You task is to drive through a traffic junction along the designated route without collision. You can not see other cars but can hear their horn sound. Horn sounds are marked in special icons on the map. Icon size indicates how certain you can hear about the horn (i.e. the probability of another car existing in that spot). You can only decide whether to proceed to the next spot on a given path or stay in the current spot.You have a maximum of 40 steps to arrive at the destination.")
OpenWindow.document.write("<h1>Control</h1>")
OpenWindow.document.write("Go: proceed to the next spot; Brake: stay in the current spot.")
OpenWindow.document.write("<h1>Contact</h1>")
OpenWindow.document.write("This is a research conducted by the University of Pittsburgh, contact researchers at hul52@pitt.edu if you have any questions.")
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
    // console.log(jsonObject);
    if (jsonObject.hasOwnProperty('attentionCheck')){attentionCheck();}
    else{
        if (jsonObject.currentTrial>20){survey();}
        else{
            displayButtons();
            draw(jsonObject);

        }
    }


};


function attentionCheck(){
    console.log('enter attention check');
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // drawBoard(ctx);
    displayButtons();
    document.getElementById('sessionReminder').innerHTML = 'This is an attention check, please click BRAKE to pass the check.'
}


function survey(){
    console.log('enter survey')
    document.getElementById('survey').style.visibility = 'visible';
    hideButtons();
    document.getElementById('sessionReminder').innerHTML = 'Please fill the following survey.'
    document.getElementsByClassName("right")[0].style.visibility = 'hidden';
    document.getElementsByClassName("middleleft")[0].style.visibility = 'hidden';
    document.getElementsByClassName("left")[0].style.visibility = 'hidden';
}


function makeid(length) {
    var result           = '';
    var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var charactersLength = characters.length;
    for ( var i = 0; i < length; i++ ) {
      result += characters.charAt(Math.floor(Math.random() *
 charactersLength));
   }
   return result;
}

function attachResults(){

    var message = new Object();
    message.type = "survey";
    var formEl = document.forms.survey;
    var formData = new FormData(formEl);
    message.helpful = formData.get('helpful');
    message.understand = formData.get('understand');
    message.satisfy = formData.get('satisfy');
    message.Post_difficulty = formData.get('Post_difficulty');
    message.Post_how = formData.get('Post_how');
    message.Post_feedback = formData.get('Post_feedback');
    message.randomCode = makeid(8);
    ws.send(JSON.stringify(message));
    document.getElementsByClassName("right")[0].style.visibility = 'hidden';
    document.getElementsByClassName("left")[0].style.visibility = 'hidden';
    document.getElementsByClassName("middleleft")[0].style.visibility = 'hidden';
    document.getElementById("survey").style.visibility = 'hidden';
    document.getElementById("ending").style.visibility = 'visible';
    document.getElementById("randomCode").innerHTML = 'Confirmation code: '+ message.randomCode;
    window.alert('Task submitted. Prolific code: 59297DCD; MTurk code: '+ message.randomCode);

}

function hideButtons(){
    document.getElementById('go').style.visibility = 'hidden';
    document.getElementById('brake').style.visibility = 'hidden';

    }

function displayButtons(){
    document.getElementById('go').style.visibility = 'visible';
    document.getElementById('brake').style.visibility = 'visible';

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

    document.getElementById('sessionReminder').innerHTML = 'Select go or stop to process to the end while avoiding collision.'
//    document.getElementById('best').innerHTML = frame_info.best.toString();
    document.getElementById('step').innerHTML = frame_info.step.toString();
    if (frame_info.hasOwnProperty('currentTrial')) {
        document.getElementById('trial').innerHTML = frame_info.currentTrial.toString();
    }


    if (frame_info.done){
        console.log('new_trial');
        drawResult(ctx,frame_info);
        setTimeout(() => {info_send();}, 2000);
    }
    else{
        //draw board
        //draw parent's vision
        if (frame_info.hasOwnProperty('humanPath')){
            // drawBoard(ctx);
            drawPath(ctx,frame_info.humanPath);
        }
        //draw player

        // draw communication
        if (frame_info.hasOwnProperty('message')) {
            for (var index in frame_info.message){
                if (index==0){continue;}
                var token = frame_info.message[index]
                if (token.hasOwnProperty('loc')){
                    for (var i =0;i<token.loc.length;i++){
                      //token.hasOwnProperty('contains_action'),frame_info.otherPaths
                        drawCarComm(ctx,token.loc[i][0],token.loc[i][1],index);

                        if (frame_info.hasOwnProperty('contains_action') && frame_info.contains_action == true){
                          console.log("Index: ");
                          console.log(index);
                          console.log(frame_info.otherPaths);
                          drawActComm(ctx,token.loc[i][0],token.loc[i][1],index,token.action,frame_info.otherPaths[index-1]);
                        }

                    }
                }
            }
        }

        if (frame_info.hasOwnProperty('players')) {
//            for (var i = 0;i<frame_info.players.length;i++){
//                var car = frame_info.players[i]
//                if (car[0]!=0||car[1]!=0){
//                    drawCar(ctx,car[0],car[1],i);
//                }
//            }
            var i = 0
            var car = frame_info.players[i]
            if (car[0]!=0||car[1]!=0){
                drawCar(ctx,car[0],car[1],i);
            }

        }


        }




}




function drawToken(ctx,x,y,text,selected){
    if(selected){ctx.strokeStyle = "red";}
    else{ctx.strokeStyle = "black";}
    ctx.font = '24px serif';
    ctx.strokeText(text,x*50+50,y*50+60);
}


function drawCar(ctx,x,y,index){
    // console.log(x,y,index);
    const image = document.getElementById('car');
    ctx.translate(x*30+45, y*30+55);
    ctx.rotate(0 * Math.PI / 180);
    if(index ==0){ctx.drawImage(image, 97, 0, 71, 37, 0, 0, 24, 12);}
    if(index ==1){ctx.drawImage(image, 0, 0, 71, 37, 0, 0, 24, 12);}
    if(index ==2){ctx.drawImage(image, 185, 0, 71, 37, 0, 0, 24, 12);}
    if(index ==3){ctx.drawImage(image, 267, 0, 71, 37, 0, 0, 24, 12);}
    if(index ==4){ctx.drawImage(image, 0, 55, 71, 37, 0, 0, 24, 12);}
    if(index ==5){ctx.drawImage(image, 95, 55, 71, 37, 0, 0, 24, 12);}
    if(index ==6){ctx.drawImage(image, 181, 55, 71, 37, 0, 0, 24, 12);}
    if(index ==7){ctx.drawImage(image, 190, 55, 71, 37, 0, 0, 24, 12);}
    if(index ==8){ctx.drawImage(image, 0, 55, 71, 37, 0, 0, 24, 12);}
    if(index ==9){ctx.drawImage(image, 267, 0, 71, 37, 0, 0, 24, 12);}
//    if (index<4){ctx.drawImage(image, index*97, 0, 65, 33, 0, 0, 24, 12);}
//    else{ctx.drawImage(image, index*95, 56, 65, 33, 0, 0, 24, 12);}
    ctx.rotate(-0 * Math.PI / 180);
    ctx.translate(-x*30-45, -y*30-55);
    ctx.strokeStyle = "red";
    ctx.font = '16px serif';
    if (index==0){ctx.strokeText('You',x*30+45,y*30+55);}
    else{ctx.strokeText(index.toString(),x*30+45,y*30+55);}
}

function drawPath(ctx,path){
    ctx.beginPath();
    ctx.setLineDash([10, 10]);
    if (path.length == 7){
      ctx.moveTo(path[0][0]*30+55, path[0][1]*30+55);
      ctx.lineTo(path[6][0]*30+55, path[6][1]*30+55);
    }
    if (path.length == 13){
        ctx.moveTo(path[0][0]*30+55, path[0][1]*30+55);
        ctx.lineTo(path[6][0]*30+55, path[6][1]*30+55);
        ctx.lineTo(path[12][0]*30+55, path[12][1]*30+55);
    }
    if (path.length == 14){
        ctx.moveTo(path[0][0]*30+55, path[0][1]*30+55);
        ctx.lineTo(path[13][0]*30+55, path[13][1]*30+55);
    }
    if (path.length == 15){
        ctx.moveTo(path[0][0]*30+55, path[0][1]*30+55);
        ctx.lineTo(path[7][0]*30+55, path[7][1]*30+55);
        ctx.lineTo(path[14][0]*30+55, path[14][1]*30+55);
    }
    ctx.strokeStyle = "red";
    ctx.stroke();
    ctx.setLineDash([]);
}


function canvas_arrow(context, fromx, fromy, tox, toy) {
  context.beginPath();
  var headlen = 10; // length of head in pixels
  var dx = tox - fromx;
  var dy = toy - fromy;
  var angle = Math.atan2(dy, dx);
  context.moveTo(fromx, fromy);
  context.lineTo(tox, toy);
  context.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
  context.moveTo(tox, toy);
  context.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
  context.stroke();
}

function drawCircle(ctx, x, y, radius, fill, stroke, strokeWidth) {
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, 2 * Math.PI, false)
  if (fill) {
    ctx.fillStyle = fill
    ctx.fill()
  }
  if (stroke) {
    ctx.lineWidth = strokeWidth
    ctx.strokeStyle = stroke
    ctx.stroke()
  }
}


function drawComm(ctx,x,y,conf){
    const image = document.getElementById('failure');
    var size = 50 *conf
    ctx.drawImage(image, x*30+55, y*30+50, size, size);
}

function drawActComm(ctx,x,y,i,action,path){
  console.log(x);
  console.log(y);
  console.log(path);
  for(var i = 0;i<path.length;i++){
    if (path[i][0] == x && path[i][1] == y) {
      console.log("FOUND PLACE IN PATH");
      console.log(action);
      console.log("\n")
      if (action == 0){
        //draw movement arrow
        if (i+1 < path.length){
          canvas_arrow(ctx,x*30+55,y*30+50,path[i+1][0]*30+50+5,path[i+1][1]*30+50)
        }
      } else {
        //draw breaking action
        drawCircle(ctx,x*30+55,y*30+50,50,'black', 'black', 2)
      }
    }
  }
  console.log("\n")

}

function drawCarComm(ctx,x,y,i){

    // const image = document.getElementById('failure');
    // var size = 50 *conf
    // ctx.drawImage(image, x*30+55, y*30+50, size, size);
    drawCar(ctx,x,y,i);
}

function drawResult(ctx,frame_info){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // drawBoard(ctx);
    for (var i = 0;i<frame_info.players.length;i++){
            var car = frame_info.players[i]
            if (car[0]!=0||car[1]!=0){
                drawCar(ctx,car[0],car[1],i);
            }
    }
    // if (frame_info.hasOwnProperty('message')) {
    //     for (var index in frame_info.message){
    //         if (index==0){continue;}
    //         var token = frame_info.message[index]
    //         if (token.hasOwnProperty('loc')){
    //             for (var i =0;i<token.loc.length;i++){
    //                 drawCarComm(ctx,token.loc[i][0],token.loc[i][1],index);
    //             }
    //         }
    //     }
    // }
    if (frame_info.results == 'success'){
        console.log('succ');

        ctx.strokeStyle = "red";
        ctx.font = '20px serif';
        ctx.strokeText('You arrived! Mission complete.',100,20);
    }
    else if (frame_info.results == 'collision'){
        console.log('collision');
        const image = document.getElementById('explosion');
        ctx.drawImage(image, frame_info.collisionLocation[0]*30+35, frame_info.collisionLocation[1]*30+30,32, 32);
        ctx.strokeStyle = "black";
        ctx.font = '20px serif';
        ctx.strokeText('Collision! Mission failed.',100,20);
    }
    else if (frame_info.results == 'timeout'){
        console.log('timeout');
        ctx.strokeStyle = "black";
        ctx.font = '20px serif';
        ctx.strokeText('Exceed maximum steps! Mission failed.',100,20);
    }
}



function drawSpace(ctx){
    ctx.arc(250,250,225,0, 2 * Math.PI);
    ctx.strokeStyle = "black";
    ctx.stroke();
}

function drawBoard(ctx){
    ctx.beginPath();
    // Box width
    var bw = 420;
    // Box height
    var bh = 420;
    // Padding
    var p = 40;

    for (var x = 0; x <= bw; x += 30) {
        if (180<=x && x<=240){
            ctx.moveTo(0.5 + x + p, p);
            ctx.lineTo(0.5 + x + p, bh + p);
        }
        else{
            ctx.moveTo(0.5 + x + p, 180+p);
            ctx.lineTo(0.5 + x + p, 240+p);
        }

    }
    for (var x = 0; x <= bh; x += 30) {
        if (180<=x && x<=240){
            ctx.moveTo(p,0.5 + x + p);
            ctx.lineTo(bh + p,0.5 + x + p);
        }
        else{
            ctx.moveTo(180+p,0.5 + x + p);
            ctx.lineTo(240+p,0.5 + x + p);
        }
    }

    ctx.strokeStyle = "black";
    ctx.stroke();
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

function action_send(key) {
  var command = new Object();
  command.player = 'car0';
  if (key == 0){
    command.command = "go" //speed up
  }
  else if (key == 1){
    command.command = "brake"
  }
  else{
    command.command = null;
  }

  var message = new Object();
  message.type = "command"
  message.message = command;
  ws.send(JSON.stringify(message));
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
  message.humanRole = 'parent'
//  state = state_update(state,message);
//  draw(state);
  ws.send(JSON.stringify(message));
}

//
//function keyDownCheck(event){
//  var e= event||window.event||arguments.callee.caller.arguments[0];
//  console.log(e.keyCode);
//  message_send(e.keyCode, true);
//}
