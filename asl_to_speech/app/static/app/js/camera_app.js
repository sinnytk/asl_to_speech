function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      // Does this cookie string begin with the name we want?
      if (cookie.substring(0, name.length + 1) === name + "=") {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

const csrftoken = getCookie("csrftoken");
var isCameraInput = false;
function toggleCapture() {
  const cap_btn = document.getElementById("capture-btn");
  const upload_btn = document.getElementById("file-upload-btn");

  if (isCameraInput) {
    cap_btn.classList.remove("hidden");
    cap_btn.classList.remove("disabled");
    upload_btn.innerHTML = "Stop video capture";
  } else {
    cap_btn.classList.add("hidden");
    cap_btn.classList.add("disabled");
    upload_btn.innerHTML = "Start video capture";
  }
}
function takePhoto() {
  const video = document.getElementById("file-image");
  var canvas = document.getElementById("rectangle");
  var context = canvas.getContext("2d");
  const width = video.videoWidth;
  const height = video.videoHeight;

  const handArea = 300;
  const pX = width / 2 - handArea / 2;
  const pY = height / 2 - handArea / 2;

  var photo = document.getElementById("photo");

  if (width && height) {
    canvas.width = handArea;
    canvas.height = handArea;
    context.drawImage(
      video,
      pX,
      pY,
      handArea,
      handArea,
      0,
      0,
      handArea,
      handArea
    );
    var data = canvas.toDataURL("image/png");
    var f = dataURLtoFile(data, "image.png");
    uploadFile(f);
    photo.setAttribute("src", data);
    photo.classList.remove("hidden");
  }
  stopWebcamStream(video);
}
function dataURLtoFile(dataurl, filename) {
  var arr = dataurl.split(","),
    mime = arr[0].match(/:(.*?);/)[1],
    bstr = atob(arr[1]),
    n = bstr.length,
    u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
}
function uploadFile(file) {
  var xhr = new XMLHttpRequest(),
    fileSizeLimit = 1024; // In MB
  if (xhr.upload) {
    // Check if file is less than x MB
    if (file.size <= fileSizeLimit * 1024 * 1024) {
      // File received / failed
      xhr.onreadystatechange = function (e) {
        if (xhr.readyState == 4) {
          // Everything is good!
          var res = JSON.parse(xhr.response);
          var annotationHtml = document.getElementsByClassName(
            "annotation-section"
          )[0];
          annotationHtml.style.display = "block";
          document.getElementsByClassName("output")[0].innerHTML =
            res.annotation;
          document.getElementById("loader").style.display = "none";
        }
      };

      // Start upload
      xhr.open(
        "POST",
        document.getElementById("file-upload-form").action,
        true
      );
      xhr.setRequestHeader("X-File-Name", file.name);
      xhr.setRequestHeader("X-CSRFToken", csrftoken);
      xhr.setRequestHeader("X-File-Size", file.size);
      var fd = new FormData();
      fd.append("image", file);
      xhr.send(fd);
    } else {
      output("Please upload a smaller file (< " + fileSizeLimit + " MB).");
    }
  }
}
function stopWebcamStream(video) {
  const stream = video.srcObject;
  stream.getTracks().forEach((track) => track.stop());
  isCameraInput = false;
  video.srcObject = null;
  toggleCapture();
}
function drawRectangle() {
  const video = document.getElementById("file-image");
  var canvas = document.getElementById("rectangle");
  const ctx = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const handArea = 300;
  const pX = canvas.width / 2 - handArea / 2;
  const pY = canvas.height / 2 - handArea / 2;

  ctx.rect(pX, pY, handArea, handArea);
  ctx.lineWidth = "6";
  ctx.strokeStyle = "red";
  ctx.stroke();

  if (isCameraInput) setTimeout(drawRectangle, 100);
}
$(document).ready(function () {
  $("#file-upload-btn").click(function () {
    var video = document.querySelector("video");
    var photo = document.getElementById("photo");
    photo.classList.add("hidden");
    photo.setAttribute("src", "");
    if (!isCameraInput) {
      annotationHtml.style.display = "none";
      document.getElementsByClassName("output")[0].innerHTML = "";
      var constraints = { audio: false, video: { width: 1280, height: 720 } };
      navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function (mediaStream) {
          video.srcObject = mediaStream;
          video.onloadedmetadata = function (e) {
            video.play().then(() => {
              isCameraInput = true;
              drawRectangle();
              toggleCapture();
              $("#capture-btn").click(function () {
                takePhoto();
              });
            });
          };
        })
        .catch(function (err) {
          console.log(err.name + ": " + err.message);
        });
    } else {
      stopWebcamStream(video);
    }
  });
});
