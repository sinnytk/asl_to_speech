let hands = null;
let camStream = null;
let handDetectionInterval = null;
let wristOriginCoord = null;

const formSelector       = '.uploader'
const formInputSelector  = '.uploader__input'
const formOutputSelector = '.uploader__output'

const feedSelector       = `${formOutputSelector} video` 

const cameraBtnSelector  = '#cameraBtn'
const captureBtnSelector = '#capBtn'


const mediaPipeCanvasSelector = '.mediapipe__canvas'
let mediaPipeCanvasCtx = null

function setOutput(output) {
  const outputBlock = $('p.output');
  outputBlock.html(output);
}

async function captureFeed() {
  if(!wristOriginCoord) {
    alert("No hand detected");
    return; 
  }
  toggleDisplay(annotationSectionSelector, true);
  let imageData = mediaPipeCanvasCtx.getImageData(wristOriginCoord[0]-80,
                                                  wristOriginCoord[1]-180, 
                                                  200, 
                                                  200);

  const imageFile = await ImageDataToBlob(imageData);
  annotateImage(imageFile)
    .then(
      (res) => {
          if(res.status === 200) {
              res.json().then(
                  (json) => {
                      setOutput(json.annotation);
                  }
              )
          }
          else {
              console.error("Something went wrong");
          }
          toggleLoader(annotationSectionSelector, false);
      },
      (err) => {
          console.error(err);
      }
  );

}

function trackHand(results) {
  const $mediaPipeCanvasEl = $(mediaPipeCanvasSelector)[0]
  
  const srcWidth = $mediaPipeCanvasEl.width
  const srcHeight = $mediaPipeCanvasEl.height

  mediaPipeCanvasCtx.save();
  mediaPipeCanvasCtx.clearRect(0, 0, srcWidth, srcHeight);
  mediaPipeCanvasCtx.drawImage(
    results.image, 0, 0, srcWidth, srcHeight);

  handLandmarks = results.multiHandLandmarks
  handedness = results.multiHandedness
  wristOriginCoord = null;
  if(handLandmarks && handedness[0].label === 'Left') {
    const wristLandmark = handLandmarks[0][0]
    wristOriginCoord = [wristLandmark.x*srcWidth, wristLandmark.y*srcHeight]
    mediaPipeCanvasCtx.beginPath();
    mediaPipeCanvasCtx.rect(wristOriginCoord[0]-80,wristOriginCoord[1]-180, 200, 200);

    mediaPipeCanvasCtx.lineWidth = "2";
    mediaPipeCanvasCtx.strokeStyle = "red";    
    mediaPipeCanvasCtx.stroke();
  } 
  mediaPipeCanvasCtx.restore();
}

function toggleCameraInput() {
  const $form       = $(formSelector)
  const $formInput  = $(formInputSelector)
  const $cameraBtn  = $(cameraBtnSelector)
  const $feed       = $(feedSelector)[0]

  if (camStream) {
    
    $form.removeClass('uploaded')
    $formInput.removeClass('is-active')
    $cameraBtn.html("Open camera feed")
    toggleDisplay(formOutputSelector, false)

    $feed.pause();
    $feed.srcObject.getTracks().forEach(a => a.stop());
    $feed.srcObject = null;
    delete camStream;
    camStream = null;

  }
  else {
    $form.addClass('uploaded')
    $formInput.addClass('is-active')
    $cameraBtn.html("Close camera feed")
    toggleDisplay(formOutputSelector, true)
    camStream = new Camera($feed, {
      onFrame: async () => {
        await hands.send({image: $feed});
      },
      width: 1200,
      height: 720
    });
    camStream.start();
  }

}

$(document).ready(function () {
  const $cameraBtn = $(cameraBtnSelector)
  const $captureBtn = $(captureBtnSelector)
  
  mediaPipeCanvasCtx = $(mediaPipeCanvasSelector)[0].getContext('2d')

  $(document).keyup(function (e) {
    // if A pressed, trigger image annotation
    // else if C pressed, toggle camera feed
    if(e.which === 65) {
      if(camStream) {
        captureFeed();
      }
    }
    else if(e.which === 67) {
      toggleCameraInput();
    }

  });
  
  $cameraBtn.on('click', function() {
    toggleCameraInput()
    this.blur();
  });

  $captureBtn.on('click', function() {
    captureFeed();
    this.blur();
  });

  hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
  }});
  
  hands.setOptions({
    maxNumHands: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  hands.onResults(trackHand);

});