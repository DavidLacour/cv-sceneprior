var timeview_canv_idx;

async function displayCrops(frame, pid,camid, numPrevFrames = 5, numFutureFrames = 5) {
    document.getElementById("crops-container").style.display = "flex";
    const cropsContainer = $("#crops-container");
    cropsContainer.empty();
    //cropsContainer.style.display = "flex";
    const currentFrame = parseInt(frame);
    document.getElementById("crops-container").innerHTML = '<button id="close-button" style="position: absolute; top: 0; left: 0;" onclick="hideCrops()">X</button>';
    $.ajax({
        method: "POST",
        url: "timeview",
        data: {
          csrfmiddlewaretoken: document.getElementsByName('csrfmiddlewaretoken')[0].value,
          personID: pid,
          frameID: parseInt(frame_str),
          viewID: camid,
          workerID: workerID,
          datasetName: dset_name
        },
        dataType: "json",
        success: function (msg) {
        async function displaycrop(msg) {
          for (let i = 0; i < msg.length; i++) {
            var box = msg[i];
            const cropFrame =box.frameID;
            if (cropFrame < 0) {
              continue;
            }
            const cropImg = await loadImage(getFrameUrl(cropFrame, "cam"+(camid+1)));
            if (cropImg !== null) {
              const canvas = createCroppedCanvas(cropImg, box, cropFrame, currentFrame);
              canvas.className = "crop-image";
              canvas.id = cropFrame;
              canvas.onclick = function() {
                if (cropFrame < frame_str) changeFrame("prev",frame_str-cropFrame);
                else
                changeFrame("next",cropFrame-frame_str);
              };
              canvas.style.maxWidth = "100px";
              cropsContainer.append(canvas);
            }
          }
        }
        displaycrop(msg);
    }

      });

  }


  
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }
  
  function createCroppedCanvas(image, box, cropFrame, currentFrame) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const cropWidth = box.x2 - box.x1;
    const cropHeight = box.y2 - box.y1;
    const ratio = 0.5;
    canvas.width = cropWidth*ratio;
    canvas.height = cropHeight*ratio;
  
    ctx.drawImage(
      image,
      box.x1,
      box.y1,
      cropWidth,
      cropHeight,
      0,
      0,
      cropWidth*ratio,
      cropHeight*ratio
    );
    
    // Add frame number text
    ctx.font = "16px Arial";
    ctx.fillStyle = "red";
    ctx.fillText(`${cropFrame}`, 5, 20);

    // Highlight the current frame with a red border
    if (cropFrame === currentFrame) {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, cropWidth*ratio, cropHeight*ratio);
    }

    return canvas;
  }
  
function getFrameUrl(frame, cameraID) {
    const frameStr = String(frame).padStart(8, "0");
    const camName = String(cameraID);
    // const url = `/static/gtm_hit/dset/${dset_name}/${undistort_frames_path}frames/${camName}/${frameStr}.jpg`;
    const url = `/static/gtm_hit/dset/${dset_name}/${undistort_frames_path}frames/${camName}/${frameStr}.jpg`;

    return url;
  }
  
  
