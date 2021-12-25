function message_send(key, flag) {
  var command = new Object();
  command.player = 0;
  if (key == 87 || key == 38){
    command.command = "thrust" //speed up
  }
  else if (key == 65 || key == 37){
    command.command = "turn_left"
  }
  else if (key == 68 || key == 39){
    command.command = "turn_right"
  }
  else if (key == 70 || key == 74){
    command.command = "fire"
  }
  else{
    command.command = null
  }
  command.isPress = flag;

  var message = new Object();
  message.type = "command"
  message.message = command;
  ws.send(JSON.stringify(message));
}

var keyCodeArry = [];

function keyDownCheck(event){
  var e= event||window.event||arguments.callee.caller.arguments[0];
  if(keyflag){
    if(keyCodeArry.indexOf(e.keyCode)!=-1){
      return;
    }
  }
  keyflag = true;
  keyCodeArry = addKeyCodeArry(e.keyCode, keyCodeArry);
  console.log(keyCodeArry);
  message_send(e.keyCode, true)
}

function keyUpCheck(event){
  keyflag = false;
  var e= event||window.event||arguments.callee.caller.arguments[0];
  keyCodeArry = deletKeyCodeArry(e.keyCode, keyCodeArry);
  console.log(keyCodeArry);
  message_send(e.keyCode, false)
}

function addKeyCodeArry(num,arr){
  var check=0;
  for (var i=0;i<arr.length;i++) {
      if (arr[i]==num) {
          check=1;
      }
  }
  if (check==0) {
    arr.push(num);
  }
  return arr;
}

function deletKeyCodeArry(num,arr){
  for (var i=0;i<arr.length;i++) {
    if (arr[i]==num) {
      arr.splice(i,1);
    }
  }
  return arr;
}


