let uploaderInputSelector = '.uploader__input'
let uploaderOutputSelector = '.uploader__output'

let annotationOutputSelector = '.output textarea'

function setOutput(output) {
    const outputBlock = $(annotationOutputSelector);
    outputBlock.html(output);
}

function handleNewVideo(videoFile) {
    toggleDisplay(uploaderOutputSelector, true); // show uploaded content div
    toggleDisplay(annotationOutputSelector, false); // hide previous output (if any)
    
    showUploadedVideo(videoFile, `${uploaderOutputSelector} video`, `${uploaderOutputSelector} p`);

    toggleDisplay(annotationSectionSelector, true); // show 
    toggleLoader(annotationSectionSelector, true);

    annotateVideo(videoFile)
        .then(
            (res) => {
                if(res.status === 200) {
                    res.json().then(
                        (json) => {
                            toggleDisplay(annotationOutputSelector, true); // show output
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

$(document).ready(function() {
    let $form    = $('#file-upload-form')
    let $inputEl = $('#file') 

    // stop default actions on all drag events
    $form.on('drag dragstart dragend dragover dragenter dragleave drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
    })
    // add class to form when dragging over
    .on('dragover dragenter', function() {
        $form.addClass('is-dragover');
    })
    // remove class from form when dragging out
    .on('dragleave dragend drop', function() {
        $form.removeClass('is-dragover');
    })
    // handle dropped image
    .on('drop', function(e) {
        const droppedVideo = e.originalEvent.dataTransfer.files[0];
        handleNewVideo(droppedVideo);
    });


    // handle manual image upload
    $inputEl.on('change', function(e) {
        const uploadedVideo = this.files[0]
        handleNewVideo(uploadedVideo);
    });
});