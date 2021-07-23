let uploaderInputSelector = '.uploader__input'
let uploaderOutputSelector = '.uploader__output'

function setOutput(output) {
    const outputBlock = $('p.output');
    outputBlock.html(output);
}

function handleNewImage(imageFile) {
    toggleDisplay(uploaderOutputSelector, true);
    setOutput("")
    
    showUploadedImage(imageFile, `${uploaderOutputSelector} img`, `${uploaderOutputSelector} p`);

    toggleDisplay(annotationSectionSelector, true);
    toggleLoader(annotationSectionSelector, true);

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
        const droppedImage = e.originalEvent.dataTransfer.files[0];
        $form.addClass('uploaded');
        handleNewImage(droppedImage);
    });


    // handle manual image upload
    $inputEl.on('change', function(e) {
        const uploadedFile = this.files[0]
        $form.addClass('uploaded');
        handleNewImage(uploadedFile);
    });
});