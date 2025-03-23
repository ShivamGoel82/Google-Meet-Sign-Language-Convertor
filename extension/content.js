function captureVideoFrame() {
    let video = document.querySelector("video");

    if (!video) return;

    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    let image = canvas.toDataURL("image/png");

    chrome.runtime.sendMessage({ action: "sendVideoFrame", image: image }, (response) => {
        if (response && response.subtitle) {
            displaySubtitles(response.subtitle);
        }
    });
}

function displaySubtitles(text) {
    let subtitleDiv = document.getElementById("asl-subtitles");

    if (!subtitleDiv) {
        subtitleDiv = document.createElement("div");
        subtitleDiv.id = "asl-subtitles";
        subtitleDiv.style.position = "absolute";
        subtitleDiv.style.bottom = "50px";
        subtitleDiv.style.left = "50%";
        subtitleDiv.style.transform = "translateX(-50%)";
        subtitleDiv.style.backgroundColor = "black";
        subtitleDiv.style.color = "white";
        subtitleDiv.style.padding = "10px";
        subtitleDiv.style.borderRadius = "5px";
        subtitleDiv.style.fontSize = "18px";
        subtitleDiv.style.zIndex = "9999";
        document.body.appendChild(subtitleDiv);
    }

    subtitleDiv.textContent = text;
}

setInterval(captureVideoFrame, 1000);
