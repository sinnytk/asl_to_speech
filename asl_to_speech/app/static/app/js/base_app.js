let annotationSectionSelector = '.annotation-section'

function showUploadedImage(file, targetImgSelector, targetCaptionSelector) {
    const targetImg     = $(targetImgSelector);
    const targetCaption = $(targetCaptionSelector);


    var reader = new FileReader();
    reader.onload = (e) => {
        targetImg.attr('src', e.target.result);
        targetCaption.html(file.name);

    }
    reader.readAsDataURL(file);
}

function showUploadedVideo(file, targetVideoSelector, targetCaptionSelector) {
    const targetImg     = $(targetVideoSelector);
    const targetCaption = $(targetCaptionSelector);


    targetImg.attr('src', window.URL.createObjectURL(file));
    targetCaption.html(file.name);
}

// Annotates an image
// Takes a File type as input
// Makes an AJAX request to server and receieves an annotation
function annotateImage(image) {
    let formData = new FormData();
    formData.append('image', image);

    return fetch('/model/annotateImage', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken() },
        body: formData
    })
}

function annotateVideo(video) {
    let formData = new FormData();
    formData.append('video', video);

    return fetch('/model/annotateVideo', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken() },
        body: formData
    })
}