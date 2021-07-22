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