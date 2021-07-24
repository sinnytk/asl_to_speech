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

// get the CSRF token from cookies
function getCsrfToken() {
    const csrftoken = getCookie("csrftoken");
    return csrftoken;
}

// Add or remove 'hidden' from element classList 
function toggleDisplay(selector, show) {
    const $elem = $(selector)
    if(show) {
        $elem.removeClass('hidden')
    }
    else {
        $elem.addClass('hidden')
    }
}

// Toggle 'loading' class from parent to show or hide loader 
function toggleLoader(parentSelector, setLoading) {
    const $elem = $(parentSelector)
    if(setLoading) {
        $elem.addClass('loading')
    } else {
        $elem.removeClass('loading')
    }
}

const ImageDataToBlob = function(imageData){
    let w = imageData.width;
    let h = imageData.height;
    
    let canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;

    let ctxx = canvas.getContext("2d");
    ctxx.putImageData(imageData, 0, 0);

    return new Promise((resolve, reject) => {
            canvas.toBlob(resolve);
    })
}
