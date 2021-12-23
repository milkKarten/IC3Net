function WebSocketTest() 
{
  if("WebSocket" in window)
  {
    alert("WebSocket is supported by your Browser!");

    // Open a websocket
    var ws = new WebSocket("ws://"+location.host+"/tsfsocket");

    ws.onopen = function()
    {
      // Web Socket is connected, send data using send()
      ws.send("start");
    };

    ws.onmessage = function(evt)
    {
      var received_msg = evt.data;
      alert(received_msg);
    };

    ws.onclose = function()
    {
      // websocket is closed.
      alert("Connection is closed...");
    };
  }
  else
  {
    // The browser doesn't support WebSocket
    alert("WebSocket NOT supported by your Browser!");
  }
}


