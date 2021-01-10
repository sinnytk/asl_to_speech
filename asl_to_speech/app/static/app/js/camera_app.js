function start_mock_translation() {
  var annotationHtml = document.getElementsByClassName("annotation-section")[0];
  annotationHtml.style.display = "block";
  setTimeout(function () {
    var sample_text = "Testing, testing, testing, mock output";
    document.getElementById("loader").style.display = "none";
    let i = 1;
    for (let el of sample_text) {
      setTimeout(function () {
        console.log(el);
        document.getElementsByClassName("output")[0].innerHTML += el;
      }, i * 1000);
      i += 1;
    }
  }, 1500);
}
$(document).ready(function () {
  $("#file-upload-btn").click(function () {
    var constraints = { audio: false, video: { width: 1280, height: 720 } };
    navigator.mediaDevices
      .getUserMedia(constraints)
      .then(function (mediaStream) {
        var video = document.querySelector("video");
        video.srcObject = mediaStream;
        video.onloadedmetadata = function (e) {
          video.play();
          start_mock_translation();
        };
      })
      .catch(function (err) {
        console.log(err.name + ": " + err.message);
      });
  });
});
