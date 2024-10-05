var rectsID = [];
var boxes = {};
var chosen_rect;
var prev_chosen_identity;
var tracklet;
var toggleTrackletClick = true;

var imgArray = [];
var arrArray = [];
var validation = {};
var identities = {};
var personID = 0;
var cameras = 7;
var camName = '';
var loadcount = 0;
var zoomOn = false;
var toggle_cuboid = true;
var toggle_unselected = true;

var undistort_frames_path ='';
var zoomratio = [];
var rotation = [50, 230, 150, 75, 265, 340, 80];
var bounds = [[0, 396, 1193, 180, 1883, 228, 1750, 1080], [0, 344, 1467, 77, 1920, 82, -1, -1],
[97, 1080, 73, 273, 864, 202, 1920, 362], [0, 444, 1920, 261, -1, -1, -1, -1],
[0, 435, 1920, 403, -1, -1, -1, -1], [0, 243, 29, 203, 656, 191, 1920, 442],
[0, 244, 1920, 162, -1, -1, -1, -1]];
var toggle_ground;
var toggle_orientation;
var to_label = 5000;

let mouseDown = false;
let selectedBox = null;
var unsavedChanges = false;
var boxesLoaded = true;

// hashsets --> rect per camera ? rect -> id to coordinates?
// store variables here? in db ? (reupload db?)
window.onload = function () {
  toggle_ground = false;
  toggle_orientation = false;

  var d = document.getElementById("changeF");
  if (d != null) {
    d.className = d.className; // + " disabled";
    if (nblabeled >= to_label) {
      var button = document.getElementById("changeF");
      button.href = "/gtm_hit/"  + dset_name + "/"+ workerID + "/processFrame";
      button.text = "Finish";
    }
  }
  if (nblabeled > 0) {
    load();
  }
  camName = cams.substring(2, cams.length - 2).split("', '");
  for (var i = 0; i < nb_cams; i++) {
    boxes[i] = {};

    arrArray[i] = new Image();
    arrArray[i].id = ("arrows" + i);
    arrArray[i].src = '/static/gtm_hit/images/arrows' + i + '_3D.png';

    imgArray[i] = new Image();
    imgArray[i].id = (i + 1);
    imgArray[i].onload = function () {
      var c = document.getElementById("canv" + this.id);
      var ctx = c.getContext('2d');

      ctx.drawImage(this, 0, 0);
      if (toggle_orientation)
        drawArrows(ctx, this.id - 1);
      c.addEventListener('contextmenu', mainClick, false);

      c.addEventListener("mousedown", onMouseDown);
      c.addEventListener("mousemove", onMouseMove);
      c.addEventListener("mouseup", onMouseUp);
      c.addEventListener("click", drawDot);
      loadcount++;
      if (loadcount == nb_cams) {
        $("#loader").hide();
      }
      update();
    }

    loadcount = 0;
    $("#loader").show();

    if (useUndistorted=="True") undistort_frames_path="undistorted_"
    // imgArray[i].src = '../../static/gtm_hit/dset/'+dset_name+'/frames/'+ camName[i]+ '/'+frame_str+'.png'; // change 00..0 by a frame variable
    imgArray[i].src = '/static/gtm_hit/dset/13apr/'+undistort_frames_path+'frames/' + camName[i] + '/' + frame_str + '.jpg'; // change 00..0 by a frame variable
    //imgArray[i].src = '../../static/gtm_hit/frames/'+ camName[i]+frame_str+'.png'; // change 00..0 by a frame variable

  }
  var topview = new Image();
  topview.id = "topviewimg";
  topview.onload = function () {
    var c = document.getElementById("topview" + this.id);
    var ctx = c.getContext('2d');
    ctx.drawImage(this, 0, 0);
    }
  topview.src = '/static/gtm_hit/dset/13apr/NewarkPennTopView2.tif'

  $(document).bind('keydown', "backspace", backSpace);

  $(document).bind('keydown', "left", leftLarge);
  $(document).bind('keydown', "right", rightLarge);
  $(document).bind('keydown', "up", upLarge);
  $(document).bind('keydown', "down", downLarge);

  $(document).bind('keydown', "a", left);
  $(document).bind('keydown', "d", right);
  $(document).bind('keydown', "w", up);
  $(document).bind('keydown', "s", down);

  $(document).bind('keydown', "i", increaseHeight);
  $(document).bind('keydown', "k", decreaseHeight);
  $(document).bind('keydown', "o", increaseWidth);
  $(document).bind('keydown', "u", decreaseWidth);
  $(document).bind('keydown', "l", increaseLength);
  $(document).bind('keydown', "j", decreaseLength);


  $(document).bind('keydown', "e", rotateCW);
  $(document).bind('keydown', "q", rotateCCW);


  $(document).bind('keydown', "tab", tab);
  $(document).bind('keydown', "space", space);
  $(document).bind("keydown", "v", validate);
  $(document).bind("keydown", "z", zoomControl);
  $(document).bind("keydown", "g", toggleGround);
  $(document).bind("keydown", "c", toggleCuboid);
  $(document).bind("keydown", "h", toggleUnselected);

  $(document).bind("keydown", "n", keyPrevFrame);
  $(document).bind("keydown", "m", keyNextFrame);

  $(document).bind("keydown", "b", toggleOrientation);
  $(document).bind("keydown", "ctrl+s", save);


  $("#pID").bind("keydown", "return", changeID);
  $("#pID").val(-1);
  $("#pHeight").val(-1);
  $("#pWidth").val(-1);

};

function onMouseDown(event) {
  mouseDown = true;
  const { offsetX, offsetY } = event;
  var mousex = offsetX * frame_size[0] / this.clientWidth;
  var mousey = offsetY * frame_size[1] / this.clientHeight;
  // Get the canvas index from the canvas id
  const canvasIndex = parseInt(event.target.id.slice(4)) - 1;
  // Check if any bounding box is selected
  let threshold = 10;
  for (const [personID, rectID] of Object.entries(rectsID)) {
    const pid = identities[rectID];
    const box = boxes[canvasIndex][pid];
    // Cuboid Base point select (Select and prep for drag)
    if (!box.cuboid || box.cuboid.length == 0) continue;
    let base_point = box.cuboid[8];

    let baseX = base_point[0];
    let baseY = base_point[1];
    if (
      mousex >= baseX - threshold &&
      mousex <= baseX + threshold &&
      mousey >= baseY - threshold &&
      mousey <= baseY + threshold
    ) {
      selectedBox = { rectID, canvasIndex }; //select for drag
      chosen_rect = rectsID.indexOf(rectID);
      break;
    }
    // Bounding Box mid point select (Select only)
    if (
      mousex >= box.xMid - threshold &&
      mousex <= box.xMid + threshold &&
      mousey >= box.y2 - threshold &&
      mousey <= box.y2 + threshold
    ) {
      chosen_rect = rectsID.indexOf(rectID);
      update();
      getTracklet();
      displayCrops(frame_str, pid,canvasIndex); //display crops --timeview.js
      timeview_canv_idx = canvasIndex;
      break;
    }
  }
  if (toggleTrackletClick) {
    // Check tracklet change frame
    const dataList = tracklet[canvasIndex];
    if (dataList==undefined) return;
    for (let i = 0; i < dataList.length; i++) {
      var x = dataList[i][1][0];
      var y  = dataList[i][1][1];
      if (
        mousex >= x - threshold &&
        mousex <= x + threshold &&
        mousey >= y - threshold &&
        mousey <= y + threshold
      ) {
        if (dataList[i][0] < frame_str) changeFrame("prev",frame_str-dataList[i][0])
        else
        changeFrame("next",dataList[i][0]-frame_str)
      }
    }
  }
}

function onMouseMove(event) {
  if (!mouseDown || !selectedBox) return;
  const { offsetX, offsetY } = event;
  var mousex = offsetX * frame_size[0] / this.clientWidth;
  var mousey = offsetY * frame_size[1] / this.clientHeight;
  const { rectID, canvasIndex } = selectedBox;
  const pid = identities[rectID];
  const box = boxes[canvasIndex][pid];
  let base_point = box.cuboid[8];
  let baseX = base_point[0];
  let baseY = base_point[1];

  const dx = mousex - baseX;
  const dy = mousey - baseY;
  // Update the box coordinates
  let newcuboid = [];
  for (let i = 0; i < 9; i++) {
    let newpoint = [box.cuboid[i][0] + dx, box.cuboid[i][1] + dy];
    newcuboid.push(newpoint);
  }
  boxes[canvasIndex][pid] = {
    ...box,
    x1: box.x1 + dx,
    x2: box.x2 + dx,
    y1: box.y1 + dy,
    y2: box.y2 + dy,
    xMid: box.xMid + dx,
    cuboid: newcuboid
  };
  update();
}

function onMouseUp() {
  if (!mouseDown || !selectedBox) return;
  const { rectID, canvasIndex } = selectedBox;
  const pid = identities[rectID];
  const box = boxes[canvasIndex][pid];
  mouseDown = false;

  $.ajax({
    method: "POST",
    url: "click",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      x: box.xMid,
      y: box.y2,
      rotation_theta: box.rotation_theta,
      object_size: box.object_size,
      canv: this.id,
      person_id: pid,
      workerID: workerID,
      datasetName: dset_name,
    },
    dataType: "json",
    success: function (msg) {
      var newrectid = msg[0].rectangleID;
      var indof = rectsID.indexOf(newrectid);
      if (indof == -1) {
        const { rectID, canvasIndex } = selectedBox;
        //save the new rectangle
        const pid = identities[rectID];
        const box = boxes[canvasIndex][pid];
        saveRect(msg, pid);
        //reassign the identity to the new rectangle
        identities[newrectid] = pid;
        delete identities[rectID];

        // reassign the rectangle to the new identity
        rectsID[rectsID.indexOf(rectID)] = newrectid;

      } else {
        chosen_rect = indof;
      }
      update();
      selectedBox = null;
    }
  });
}

function mainClick(e) {
  e.preventDefault();
  const { offsetX, offsetY } = e;
  var xCorr = Math.round(offsetX * frame_size[0] / this.clientWidth);
  var yCorr = Math.round(offsetY * frame_size[1] / this.clientHeight);
  if (zoomOn)
    zoomOut();
  //post
  $.ajax({
    method: "POST",
    url: "click",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      x: xCorr,
      y: yCorr,
      canv: this.id,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      var rid = msg[0].rectangleID;
      var indof = rectsID.indexOf(rid);
      if (indof == -1) {
        rectsID.push(rid);
        chosen_rect = rectsID.length - 1;
        const personID = msg[0].personID;
        identities[rid] = personID;
        validation[personID] = true;
        saveRect(msg, personID);
      } else {
        chosen_rect = indof;
      }
      update();
    }
  });
}

function getTracklet(e) {
  if(e)
  e.preventDefault();
  
  var pid = identities[rectsID[chosen_rect]];
  $.ajax({
    method: "POST",
    url: "tracklet",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      personID: pid,
      frameID: parseInt(frame_str),
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      tracklet = msg;
      for (var i = 0; i < nb_cams; i++) {
        var c = document.getElementById("canv" + (i + 1));
        var ctx = c.getContext("2d");
        ctx.strokeStyle = "chartreuse";
        ctx.lineWidth = "2";
        ctx.strokeStyle = "red";
        ctx.font = "11px Arial";
        ctx.fillStyle = "red";
        const dataList = msg[i];
        if (dataList==undefined) continue;
        ctx.beginPath();
        ctx.moveTo(dataList[0][1][0], dataList[0][1][1]);
        ctx.fillText(dataList[0][0], dataList[0][1][0], dataList[0][1][1] - 5);
        for (let i = 1; i < dataList.length; i++) {
          ctx.lineTo(dataList[i][1][0], dataList[i][1][1]);
          ctx.fillText(dataList[i][0], dataList[i][1][0], dataList[i][1][1] - 5);
        }
        ctx.stroke();
        ctx.closePath()
        
        if (toggleTrackletClick) {
          for (let i = 1; i < dataList.length; i++) {
              ctx.beginPath();
              ctx.fillStyle = "green";
              ctx.fillRect(dataList[i][1][0] - 3, dataList[i][1][1] - 3, 6, 6);
              ctx.stroke();
              ctx.closePath();
          }
        }
        
      }
    }
  });
}

function interpolate(e) {
  if(e)
  e.preventDefault();
  
  var pid = prev_chosen_identity || identities[rectsID[chosen_rect]];
  $.ajax({
    method: "POST",
    url: "interpolate",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      personID: pid,
      frameID: parseInt(frame_str)-1,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      alert(msg["message"])
      loader_db("load")
    },
    error: function (msg) {
      console.log("Error while interpolating,running copy from previous/next frame")
      showCopyBtn()
    }
  });
}

function showCopyBtn(){
  var copyBtn = document.getElementById('copyBtn');
  var pid = prev_chosen_identity || identities[rectsID[chosen_rect]];
  copyBtn.innerHTML = "Copy Prev/Next (ID:"+pid+")";
  if (copyBtn.style.display === 'none') {
    copyBtn.style.display = 'inline';
  }
}
function copyPrevOrNext(e) {
  copyBtn.style.display = 'none';
  if(e)
  e.preventDefault();
  var pid = prev_chosen_identity || identities[rectsID[chosen_rect]];
  $.ajax({
    method: "POST",
    url: "copy",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      personID: pid,
      frameID: parseInt(frame_str),
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      console.log(msg["message"])
      loader_db("load")
    },
    error: function (msg) {
      console.log("Error while copying from previous/next frame")
    }
  });
}

function createVideo(e) {
  $.ajax({
    method: "POST",
    url: "createvideo",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      alert("Video created.")
    },
    error: function (msg) {
      alert("Error while creating video.")
    }
  });
}

function removeCompleteFlags(e) {
  $.ajax({
    method: "POST",
    url: "resetacflags",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      loader_db("load");
      alert("AC flags removed.")
    },
    error: function (msg) {
    }
  });
}

function backSpace() {
  if (rectsID.length > 0) {
    var rid = rectsID[chosen_rect];
    rectsID.splice(chosen_rect, 1);
    var idPers = identities[rid];
    delete validation[idPers];
    delete identities[rid];
    //validation_dict.pop(idPers)
    //identities.pop(idRect)
    //for i in range(NB_PICTURES):
    //  if idPers in person_rect[i]:
    //    person_rect[i].pop(idPers)
    if (chosen_rect == rectsID.length) {
      chosen_rect--;
    }
    if (zoomOn) {
      zoomOut();
    }
    update();
  }
  return false;
}

function tab() {
  if (rectsID.length <= 1)
    return false;

  chosen_rect++;
  if (zoomOn)
    zoomOut();
  update();
  getTracklet();
  return false;
}

function keyNextFrame() {
  changeFrame('next',parseInt(frame_inc))
}

function keyPrevFrame() {
  changeFrame('prev',parseInt(frame_inc))
}
function space() {
  if (rectsID.length <= 1)
    return false;

  chosen_rect--;
  if (zoomOn)
    zoomOut();
  update();
  return false;
}

function sendAction(action) {
  const box = boxes[0][identities[rectsID[chosen_rect]]];
  const data = {
    "action": action,
    "Xw": box["Xw"],
    "Yw": box["Yw"],
    "Zw": box["Zw"],
    "rotation_theta": box["rotation_theta"],
    "object_size": box["object_size"],
  };
  sendAJAX("action", JSON.stringify(data), rectsID[chosen_rect], rectAction,false);
  update();
  return false;
}

function left() {
  return sendAction({ "move": "left" });
}
function right() {
  return sendAction({ "move": "right" });
}
function up() {
  return sendAction({ "move": "up" });
}
function down() {
  return sendAction({ "move": "down" });
}

function leftLarge() {
  return sendAction({ "move": "left","stepMultiplier":10});
}
function rightLarge() {
  return sendAction({ "move": "right","stepMultiplier":10 });
}
function upLarge() {
  return sendAction({ "move": "up","stepMultiplier":10 });
}
function downLarge() {
  return sendAction({ "move": "down","stepMultiplier":10 });
}

function rotateCW() {
  return sendAction({ "rotate": "cw"});
}
function rotateCCW() {
  return sendAction({ "rotate": "ccw" });
}
function increaseHeight() {
  return sendAction({ "changeSize": { "height": "increase" } });
}
function decreaseHeight() {
  return sendAction({ "changeSize": { "height": "decrease" } });
}
function increaseWidth() {
  return sendAction({ "changeSize": { "width": "increase" } });
}
function decreaseWidth() {
  return sendAction({ "changeSize": { "width": "decrease" } });
}
function increaseLength() {
  return sendAction({ "changeSize": { "length": "increase" } });
}
function decreaseLength() {
  return sendAction({ "changeSize": { "length": "decrease" } });
}

// function changeSize(changeWidth, changeHeight, increase) {
//   var ind = getIndx();
//   var pid = identities[rectsID[chosen_rect]];
//   var rect = boxes[ind][pid];
//   var delta = increase ? 1 : -1;

//   if (changeWidth) {
//     rect.x1 -= delta;
//     rect.x2 += delta;

//     if (rect.x2 - rect.x1 < 1) {
//       rect.x1 = rect.x2 - 1;
//     }

//     var widthSize = (rect.x2 - rect.x1) * rect.ratio;
//     updateSize(false, widthSize, ind);
//   }

//   if (changeHeight) {
//     rect.y1 += delta;

//     if (rect.y2 - rect.y1 < 1) {
//       rect.y1 = rect.y2 - 1;
//     }

//     var heightSize = (rect.y2 - rect.y1) * rect.ratio;
//     updateSize(true, heightSize, ind);
//   }

//   boxes[ind][pid] = rect;
//   return false;
// }


// function incrWidth() {
//   var ind = getIndx();
//   var pid = identities[rectsID[chosen_rect]];
//   var rect = boxes[ind][pid];
//   rect.x1 = rect.x1-1;
//   rect.x2 = rect.x2+1;
//   boxes[ind][pid] = rect;

//   var size = (rect.x2-rect.x1)*rect.ratio;
//   updateSize(false,size,ind);
//   return false;

// }

// function decrWidth() {
//   var ind = getIndx();
//   var pid = identities[rectsID[chosen_rect]];
//   var rect = boxes[ind][pid];
//   rect.x1 = rect.x1+1;
//   if(rect.x2 - rect.x1 < 1) {
//     rect.x1 = rect.x2 -1;
//   } else {
//     rect.x2 = rect.x2-1;
//   }
//   boxes[ind][pid] = rect;
//   var size = (rect.x2-rect.x1)*rect.ratio;
//   updateSize(false,size,ind);
//   return false;

// }

// function incrHeight() {

//   var ind = getIndx();
//   var pid = identities[rectsID[chosen_rect]];
//   var rect = boxes[ind][pid];
//   rect.y1 = rect.y1-1;
//   boxes[ind][pid] = rect;

//   var size = (rect.y2-rect.y1)*rect.ratio;
//   updateSize(true,size,ind);
//   return false;

// }

// function decrHeight() {
//   var ind = getIndx();
//   var pid = identities[rectsID[chosen_rect]];
//   var rect = boxes[ind][pid];
//   rect.y1 = rect.y1+1;
//   if(rect.y2 - rect.y1 < 1) {
//     rect.y1 = rect.y2-1;
//   }
//   boxes[ind][pid] = rect;

//   var size = (rect.y2-rect.y1)*rect.ratio;
//   updateSize(true,size,ind);
//   return false;
// }

function getIndx() {
  var h = -1;
  var retInd = -1;
  var pid = identities[rectsID[chosen_rect]];
  for (var i = 0; i < nb_cams; i++) {
    r = boxes[i][pid];
    tpH = Math.abs(r.y1 - r.y2);

    if (tpH > h) {
      h = tpH;
      retInd = i;
    }
  }
  return retInd;
}

function updateSize(height, size, ind) {
  var r = rectsID[chosen_rect];
  var pid = identities[r];
  for (var i = 0; i < nb_cams; i++) {
    rect = boxes[i][pid];
    if (i != ind && rect.y1 != 0) {
      if (height) {
        var b = Math.round(rect.y2 - size / rect.ratio);
        if (rect.y2 - b < 1)
          b = rect.y2 - 1;
        rect.y1 = b;
      } else {
        var delta = size / (2 * rect.ratio);
        var c = Math.round(rect.xMid + delta);
        var a = Math.round(rect.xMid - delta);
        if (c - a < 1)
          a = c - 1;
        rect.x1 = a;
        rect.x2 = c;
      }
    }
    boxes[i][pid] = rect;
  }
  update()
}

function save(e) {
  if (e) e.preventDefault();
  var dims = [];
  var k = 0;
  for (var i = 0; i < rectsID.length; i++) {
    var rid = rectsID[i];
    var pid = identities[rid];
    let box = boxes[0][pid];
    box["personID"] = pid;
    dims.push(box);
  }
  $.ajax({
    method: "POST",
    url: 'save',
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      data: JSON.stringify(dims),
      ID: frame_str,
      workerID: workerID,
      datasetName: dset_name
    },
    success: function (msg) {
      console.log(msg);
      unsavedChanges = false;
      $("#unsaved").html("All changes saved.");
    }
  });

}

function saveCurrentlySelected() {
  var dims = [];
  var pid = identities[rectsID[chosen_rect]];
  let box = boxes[0][pid];
  if (!box) return;
  box["personID"] = pid;
  dims.push(box);

  $.ajax({
    method: "POST",
    url: 'save',
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      data: JSON.stringify(dims),
      ID: frame_str,
      workerID: workerID,
      datasetName: dset_name
    },
    success: function (msg) {
      console.log(msg);
      unsavedChanges = false;
      $("#unsaved").html("All changes saved.");
    }
  });
}

function load() {
  loader_db('load');
}

function load_prev() {
  loader2('loadprev');

}

function loader(uri) {

  $.ajax({
    method: "POST",
    url: uri,
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      ID: frame_str,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: 'json',
    success: function (msg) {
      boxesLoaded=false;
      clean();
      var maxID = 0;
      for (var i = 0; i < msg.length; i++) {
        var rid = msg[i][0].rectangleID;
        var indof = rectsID.indexOf(rid);
        if (indof == -1) {
          rectsID.push(rid);
          saveRectLoad(msg[i]);
          chosen_rect = rectsID.length - 1;
          identities[rid] = msg[i][nb_cams];
          var pid = msg[i][nb_cams];
          if (pid > maxID)
            maxID = pid;
          if (uri == "loadprev")
            validation[pid] = false;
          else
            validation[pid] = msg[i][parseInt(nb_cams) + 1];
        }
      }
      personID = maxID + 1;
      update();
      $("#unsaved").html("All changes saved.");
      unsavedChanges = false;
      boxesLoaded=true;
    },
    error: function (msg) {
      if (uri == "load")
        load_prev();
    }
  });
}

function loader2(uri) {

  $.ajax({
    method: "POST",
    url: uri,
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      ID: frame_str,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: 'json',
    success: function (msg) {
      clean();
      var maxID = 0;
      for (var i = 0; i < msg.length; i++) {
        const box = msg[i];
        var rid = box.rectangleID;
        var indof = rectsID.indexOf(rid);
        if (indof == -1) {
          rectsID.push(rid);
          var pid = box.personID;
          identities[rid] = pid;
          sendAJAX("action", JSON.stringify(box), rid, rectAction,true);
          if (pid > maxID)
            maxID = pid;
          if (uri == "loadprev")
            validation[pid] = false;
          else
            validation[pid] = true;
        }
      }
      personID = maxID + 1;
      update();
      $("#unsaved").html("All changes saved.");
    },
    error: function (msg) {
      if (uri == "load")
        load_prev();
    }
  });
}

function loader_db(uri) {
  $.ajax({
    method: "POST",
    url: uri,
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      ID: frame_str,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: 'json',
    success: function (msg) {
      boxesLoaded=false;
      clean();
      var maxID = 0;
      for (var i = 0; i < msg[0].length; i++) {
        var rid = msg[0][i].rectangleID;
        var indof = rectsID.indexOf(rid);
        if (indof == -1) {
          rectsID.push(rid);
          chosen_rect = rectsID.length - 1;
        }else{
          chosen_rect = rectsID[indof];
          var pid = msg[0][i].person_id
          identities[rid] = pid;
        }
        var pid = msg[0][i].person_id
        identities[rid] = pid;
        for (var cami = 0; cami < nb_cams; cami++) {
          boxes[cami][pid] = msg[cami][i];
          }
        if (pid > maxID)
          maxID = pid;
        if (uri == "loadprev")
          validation[pid] = false;
        else
          validation[pid] = true;
          
      }

      if (prev_chosen_identity!=undefined){
        if (prev_chosen_identity in boxes[0]){
          chosen_rect =  rectsID.indexOf(boxes[0][prev_chosen_identity].rectangleID)
          getTracklet();
          displayCrops(frame_str, prev_chosen_identity, timeview_canv_idx); //display crops --timeview.js
          showCopyBtn()
        }
        else {
          interpolate()
        }
      }
      
      personID = maxID + 1;
      boxesLoaded=true;
      $("#unsaved").html("All changes saved.");
      update();
    },
    error: function (msg) {
      if (uri == "load")
        load_prev();
    }
  });
}

function clean() {
  for (var i = 0; i < nb_cams; i++) {
    boxes[i] = {};
    rectsID = [];
    validation = {};
    identities = {};
    personID = 0;
    chosen_rect = 0;
  }
  update();
}


function changeFrame(order, increment) {
  if(boxesLoaded) saveCurrentlySelected();
  if (nblabeled >= to_label) {
    return true;
  }
  boxesLoaded=false;
  $.ajax({
    method: "POST",
    url: 'changeframe',
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      order: order,
      frameID: frame_str,
      incr: increment,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      frame_str = msg['frame'];
      nblabeled = msg['nblabeled'];
      if (nblabeled >= to_label) {
        var button = document.getElementById("changeF");
        button.href = "/gtm_hit/" + dset_name + "/"+ workerID + "/processFrame";
        button.text = "Finish";
      }
      loadcount = 0;
      $("#loader").show();
      fstr = parseInt(frame_str);
      $("#frameID").html("Frame ID: " + fstr.toString() + "&nbsp;&nbsp;");
      for (var i = 0; i < nb_cams; i++)
        imgArray[i].src = '/static/gtm_hit/dset/' + dset_name + '/'+undistort_frames_path+'frames/' + camName[i] + '/' + frame_str + '.jpg'; // change 00..0 by a frame variable
      //imgArray[i].src = '../../static/gtm_hit/frames/'+ camName[i]+frame_str+'.png'; // change 00..0 by a frame variable

    },

    complete: function (msg) {
      prev_chosen_identity= identities[rectsID[chosen_rect]];
      clean()
      load();
      showCopyBtn()
    }
  });

}

function next() {
  changeFrame('next', 1);
}

function prev() {
  changeFrame('prev', 1);
}

function nextI() {
  changeFrame('next', 10);
}

function prevI() {
  changeFrame('prev', 10);
}

function validate() {
  var rid = rectsID[chosen_rect];
  var idPers = identities[rid];
  validation[idPers] = true;
  return false;
}

// function changeID() {
//   var newID = parseInt($("#pID").val());
//   if (rectsID.length > 0 && newID >= 0) {
//     var rid = rectsID[chosen_rect];
//     var pid = identities[rid];
//     var match = false;
//     for (key in identities) {
//       if (identities[key] == newID)
//         match = true;
//     }
//     if (!match) {
//       validation[newID] = validation[pid];
//       delete validation[pid];
//       identities[rid] = newID;
//       for (key in boxes) {
//         if (pid in boxes[key]) {
//           var args = boxes[key][pid];
//           boxes[key][newID] = args;
//           delete boxes[key][pid];
//         }
//       }
//       $("#pID").val(newID);
//     } else {
//       $("#pID").val(pid);
//     }
//   }
// }

function changeID(opt) {
  const propagateElement = document.getElementById('propagate');
  const conflictsElement = document.getElementById('conflicts');
  const propagateValue = propagateElement.value;
  const conflictsValue = conflictsElement.value;

  var newID = parseInt($("#pID").val());
  if (opt==undefined)opt="";
  const old_chosen_rect = chosen_rect;
  $.ajax({
    method: "POST",
    url: "changeid",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      newPersonID: newID,
      frameID: parseInt(frame_str),
      personID: identities[rectsID[chosen_rect]],
      workerID: workerID,
      datasetName: dset_name,
      options: JSON.stringify({'propagate':propagateValue,'conflicts':conflictsValue})
    },
    dataType: "json",
    success: function (msg) {
      loader_db('load');
      $("#pID").val(newID);
      chosen_rect = old_chosen_rect;
    }
  });
}

function personAction(opt) {
  if (opt==undefined)return false;
  const old_chosen_rect = chosen_rect;
  prev_chosen_identity = identities[rectsID[chosen_rect]];
  $.ajax({
    method: "POST",
    url: "person",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      personID: identities[rectsID[chosen_rect]],
      workerID: workerID,
      datasetName: dset_name,
      options: JSON.stringify(opt)
    },
    dataType: "json",
    success: function (msg) {
      loader_db('load');
      if (!opt["delete"] )chosen_rect = old_chosen_rect;
      
    }
  });
}


function sendAJAX(uri, data, id, suc, load) {
  $.ajax({
    method: "POST",
    url: uri,
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      data: data,
      ID: id,
      workerID: workerID,
      datasetName: dset_name
    },
    dataType: "json",
    success: function (msg) {
      if (msg.length > 0)
        suc(msg, id, load);
      update();
    }
  });
}

function saveRect(msg, pid) {
  for (var i = 0; i < msg.length; i++) {
    var ind = msg[i].cameraID;
    boxes[ind][pid] = msg[i];
  }
}

function saveRectLoad(msg) {
  for (var i = 0; i < msg.length - 2; i++) {
    var ind = msg[i].cameraID;
    boxes[ind][msg[nb_cams]] = msg[i];
  }
}

function rectAction(msg, id, load) {
  var pid = identities[id];
  // if(typeof boxes[0][pid] == "undefined") {
  //   return false;
  // }
  if (typeof pid == "undefined") {
    return false;
  }
  var index = rectsID.indexOf(id);
  var nextRect = msg[0].rectangleID;
  rectsID.splice(index, 1);
  rectsID.push(nextRect);
  chosen_rect = rectsID.length - 1;
  if (load && prev_chosen_identity!=undefined){
    if (prev_chosen_identity in boxes[0])
      chosen_rect =  rectsID.indexOf(boxes[0][prev_chosen_identity].rectangleID)
  }
  identities[nextRect] = pid;
  if (nextRect != id) delete identities[id];
  validation[pid] = true;

  for (var i = 0; i < msg.length; i++) {
    var f = msg[i];
    var ind = f.cameraID;
    var oldRect = boxes[ind][pid];

    var newRect = msg[i];
    // var heightR = Math.abs(oldRect.y1-oldRect.y2)*oldRect.ratio;
    // var widthR = Math.abs(oldRect.x1-oldRect.x2)*oldRect.ratio;

    // if(newRect.ratio > 0){
    //   newRect.y1 = Math.round(newRect.y2 - (heightR/newRect.ratio));
    //   var delta = widthR/(2*newRect.ratio);
    //   newRect.x2 = Math.round(newRect.xMid + delta);
    //   newRect.x1 = Math.round(newRect.xMid - delta);
    // }
    boxes[ind][pid] = newRect;
  }
}

function update() {
  tracklet = null;
  if (chosen_rect==undefined) chosen_rect = 0;
  chosen_rect = ((chosen_rect % rectsID.length) + rectsID.length) % rectsID.length;
  $("#pID").val(identities[rectsID[chosen_rect]]);

  drawRect();
  if (toggle_ground)
    drawGround();

  var d = document.getElementById("changeF");
  if (d != null) {
    if (rectsID.length > 1) {
      if (checkRects())
        d.className = d.className.replace(" disabled", "");
      else if (d.className.indexOf("disabled") == -1)
        d.className = d.className + " disabled";
    } else if (d.className.indexOf("disabled") == -1) {
      d.className = d.className;// + " disabled";
    }
  }
  unsavedChanges = true;
  $("#unsaved").html("Unsaved changes.");
}


function drawDot(event) {
  const { offsetX, offsetY } = event;
  var x = offsetX * frame_size[0] / this.clientWidth;
  var y = offsetY * frame_size[1] / this.clientHeight;
  const dotRadius = 5;
  var c = event.currentTarget
  var ctx = c.getContext("2d");
  ctx.beginPath();
  ctx.arc(x, y, 2, 0, 2 * Math.PI);
  ctx.fillStyle = "black";
  ctx.fill();
  ctx.closePath();

}

function drawLine(ctx, v1, v2) {
  ctx.beginPath();
  ctx.strokeStyle = "pink";
  ctx.lineWidth = "2";
  ctx.moveTo(v1[0], v1[1]);
  ctx.lineTo(v2[0], v2[1]);
  ctx.stroke();
  ctx.closePath();
}


function drawCuboid(ctx, vertices) {
  // Draw lines for the base rectangle
  drawLine(ctx, vertices[0], vertices[1]);
  drawLine(ctx, vertices[1], vertices[3]);
  drawLine(ctx, vertices[2], vertices[3]);
  drawLine(ctx, vertices[2], vertices[0]);

  // Draw lines for the top rectangle
  drawLine(ctx, vertices[4], vertices[5]);
  drawLine(ctx, vertices[5], vertices[7]);
  drawLine(ctx, vertices[6], vertices[7]);
  drawLine(ctx, vertices[6], vertices[4]);

  // Draw lines connecting the base and top rectangles
  drawLine(ctx, vertices[0], vertices[4]);
  drawLine(ctx, vertices[1], vertices[5]);
  drawLine(ctx, vertices[2], vertices[6]);
  drawLine(ctx, vertices[3], vertices[7]);

  //draw direction
  if (vertices.length>8){
  drawLine(ctx, vertices[8], vertices[9]);

  // mark the base point
  ctx.beginPath();
  ctx.fillStyle = "red";
  ctx.fillRect(vertices[8][0] - 5, vertices[8][1] - 5, 10, 10);
  ctx.stroke();
  ctx.closePath();
  }
}

function removePersonFromAll() {
  const isConfirmed = confirm("Are you sure you want to delete this object from all frames? This cannot be undone.");
  if (isConfirmed) {
    personAction({'delete':true});
  }
}


function drawRect() {
  for (var i = 0; i < nb_cams; i++) {
    var c = document.getElementById("canv" + (i + 1));
    var ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    //check if image is loaded
    if (!imgArray[i].complete || imgArray[i].naturalWidth === 0) continue;
    ctx.drawImage(imgArray[i], 0, 0);
    if (toggle_orientation)
      drawArrows(ctx, i);
  }
  var heightR = 0;
  var widthR = 0;
  var sumH = 0;
  for (key in boxes) {
    for (var r = 0; r < rectsID.length; r++) {
      var field = boxes[key][identities[rectsID[r]]];
      if (field.y1 != -1 && field.y2 != -1 && field.x1 != -1) {
        var c = document.getElementById("canv" + (field.cameraID + 1));
        var ctx = c.getContext("2d");
        
        //show only selected 
        if (!(r == chosen_rect) && !toggle_unselected) continue;
        //draw cuboid
        if (toggle_cuboid && field.cuboid && field.cuboid.length!=0) drawCuboid(ctx, field.cuboid);

        var w = field.x2 - field.x1;
        var h = field.y2 - field.y1;
        if (r == chosen_rect) {
          ctx.strokeStyle = "cyan";
          ctx.lineWidth = "3";
          heightR += (field.y2 - field.y1) * field.ratio;
          widthR += (field.x2 - field.x1) * field.ratio;
          sumH += 1;
        } else {
          var pid = identities[field.rectangleID];
          if (validation[pid])
            ctx.strokeStyle = "white";
          else
            ctx.strokeStyle = "yellow";

          if (field.annotation_complete) ctx.strokeStyle = "green";
          ctx.lineWidth = "4";
        }
        
        

        ctx.beginPath();
        ctx.rect(field.x1, field.y1, w, h);

        ctx.stroke();
        ctx.closePath();

        ctx.beginPath();
        ctx.fillStyle = "green";
        ctx.fillRect(field.xMid - 5, field.y2 - 5, 10, 10);
        ctx.stroke();
        ctx.closePath();

        ctx.beginPath();
        ctx.fillStyle = "black";
        if (field.annotation_complete) ctx.fillStyle = "green";
        ctx.fillRect(field.x1, field.y1 - 27, 50, 20);
        ctx.stroke();
        ctx.closePath();

        if (r == chosen_rect) {
          ctx.fillStyle = "cyan";
        } else {
          ctx.fillStyle = "white";
          
        }
        ctx.font = "20px Arial";

        ctx.fillText("ID:" + identities[field.rectangleID], field.x1, field.y1 - 10);
      }
    }
  }
  if (chosen_rect >= 0) {
    let pid = identities[rectsID[chosen_rect]];
    let box = boxes[key][pid];
    //round to 2 decimal places
    $("#pHeight").text("Height: "+box.object_size[0].toFixed(3)); 
    $("#pWidth").text("Width: "+box.object_size[1].toFixed(3));
    $("#pLength").text("Length: "+box.object_size[2].toFixed(3));
    $("#pID").text(pid);


  } else {
    $("#pHeight").text(-1);
    $("#pWidth").text(-1);
    $("#pLength").text(-1);
    $("#pID").text(-1);
  }

}

function drawGround() {
  for (var i = 0; i < nb_cams; i++) {
    var c = document.getElementById("canv" + (i + 1));
    var ctx = c.getContext("2d");
    ctx.strokeStyle = "chartreuse";
    ctx.lineWidth = "2";
    ctx.beginPath();

    ctx.moveTo(bounds[i][0], bounds[i][1]);
    for (var j = 2; j < bounds[i].length; j = j + 2) {
      if (bounds[i][j] >= 0) {
        ctx.lineTo(bounds[i][j], bounds[i][j + 1]);
      }
    }
    ctx.stroke();
    ctx.closePath();

  }
}

function drawArrows(ctx, idx) {
  ctx.drawImage(arrArray[idx], 0, 0);
}

function zoomControl() {
  if (rectsID.length > 0) {
    if (!zoomOn) {
      zoomIn();
    } else {
      zoomOut();
    }

  }
  update();
}

function isBoundingBoxInCanvas(box, canvas) {
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;

  const boxLeft = Math.min(box.x1, box.x2);
  const boxRight = Math.max(box.x1, box.x2);
  const boxTop = Math.min(box.y1, box.y2);
  const boxBottom = Math.max(box.y1, box.y2);

  const isBoxLeftInCanvas = boxLeft >= 0 && boxLeft < canvasWidth;
  const isBoxRightInCanvas = boxRight > 0 && boxRight <= canvasWidth;
  const isBoxTopInCanvas = boxTop >= 0 && boxTop < canvasHeight;
  const isBoxBottomInCanvas = boxBottom > 0 && boxBottom <= canvasHeight;

  return (isBoxLeftInCanvas || isBoxRightInCanvas) &&
    (isBoxTopInCanvas || isBoxBottomInCanvas);
}

function zoomIn() {
  for (var i = 0; i < nb_cams; i++) {
    var pid = identities[rectsID[chosen_rect]];
    var r = boxes[i][pid];

    var c = document.getElementById("canv" + (i + 1));
    if (isBoundingBoxInCanvas(r, c)) { zoomratio[i] = c.height * 60 / (100 * (r.y2 - r.y1)); }
    else {
      zoomratio[i] = null;
      continue;
    }
    if (zoomratio[i] != Infinity) {

      var ctx = c.getContext('2d');
      c.width = c.width / zoomratio[i];
      c.height = c.height / zoomratio[i];
      var originx = r.xMid - c.width / 2;
      // var originx = r.xMid;
      var originy = r.y1 - 12.5 * c.clientHeight / 100;
      // ctx.scale(1.75,1.75);
      ctx.translate(-originx, -originy);

    }
  }
  zoomOn = true;
  return false;

}

function zoomOut() {
  for (var i = 0; i < nb_cams; i++) {
    var c = document.getElementById("canv" + (i + 1));
    if (zoomratio[i] != undefined && zoomratio[i] != Infinity) {
      c.width = c.width * zoomratio[i];
      c.height = c.height * zoomratio[i];
    }
  }
  zoomOn = false;
  return false;
}


function toggleGround() {
  if (toggle_ground == false)
    toggle_ground = true;
  else
    toggle_ground = false;
  update();
  return false;
}

function toggleCuboid() {
  if (toggle_cuboid == false)
    toggle_cuboid = true;
  else
    toggle_cuboid = false;
  update();
  return false;
}
function toggleUnselected() {
  if (toggle_unselected == false)
    toggle_unselected = true;
  else
    toggle_unselected = false;
  update();
  return false;
}

function toggleOrientation() {
  if (toggle_orientation == false)
    toggle_orientation = true;
  else
    toggle_orientation = false;
  update();
  return false;
}

function checkRects() {
  return true
  var c = 0;
  for (var i = 0; i < rectsID.length; i++) {
    var personID = identities[rectsID[i]];
    if (validation[personID])
      c++;
  }
  if (c > 1)
    return true;
  else
    return false;
}


function load_file(f) {
  var re = f.match(/_(.*)\./);
  if (re == null)
    var frame_string = f.split(".")[0];
  else
    var frame_string = f.match(/_(.*)\./).pop();
  $.ajax({
    method: "POST",
    url: "loadfile",
    data: {
      csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
      ID: f
    },
    dataType: 'json',
    success: function (msg) {
      clean();
      load_frame(frame_string);
      var maxID = 0;
      for (var i = 0; i < msg.length; i++) {
        var rid = msg[i][0].rectangleID;
        var indof = rectsID.indexOf(rid);
        if (indof == -1) {
          rectsID.push(rid);
          saveRectLoad(msg[i]);
          chosen_rect = rectsID.length - 1;
          identities[rid] = msg[i][7];
          var pid = msg[i][7];
          if (pid > maxID)
            maxID = pid;

          validation[pid] = msg[i][8];
        }
        personID = maxID + 1;
        update();
      }
    }
  });

}

async function load_frame(frame_string) {
  loadcount = 0;
  $("#loader").show();
  fstr = frame_string;
  fstr = fstr.replace(/^0*/, "");
  frame_str = frame_string;
  $("#frameID").html("Frame ID: " + fstr + "&nbsp;&nbsp;");
  for (var i = 0; i < cameras; i++)
    //imgArray[i].src = '../../static/marker/day_2/annotation_final/'+ camName[i]+ '/begin/'+frame_string+'.png'; // change 00..0 by a frame variable
    // imgArray[i].src = '../../static/gtm_hit/dset/rayon4/frames/'+ camName[i]+"/"+frame_str+'.png'; // change 00..0 by a frame variable
    
    //imgArray[i].src = '../../static/gtm_hit/dset/invision/'+undistort_frames_path+'frames/' + camName[i] + "/" + frame_str + '.jpg'; // change 00..0 by a frame variable
    //var imgSrc = '/static/gtm_hit/dset/'+dset_name+'/'+undistort_frames_path+'frames/' + camName[i] + "/" + frame_str + '.jpg';
    var imgSrc = '/static/gtm_hit/dset/13apr/'+undistort_frames_path+'frames/' + camName[i] + "/" + frame_str + '.jpg';
    const loadedImg = await loadImage(imgSrc);
    if (loadedImg !== null) {
      imgArray[i].src = imgSrc;
    }
  clean();
  update();
}
