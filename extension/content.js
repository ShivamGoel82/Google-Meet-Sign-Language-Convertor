console.log("Content script injected.");

// Function to fetch subtitle from backend
function fetchSubtitle() {
    chrome.runtime.sendMessage({ type: "fetch_subtitle" }, (response) => {
        if (chrome.runtime.lastError) {
            console.warn("Extension context invalidated. Attempting to reconnect...");
            reconnectExtension();  // Try to reconnect instead of reloading
            return;
        }
        if (response && response.text) {
            updateSubtitle(response.text);
        }
    });
}

// Function to display subtitles
function updateSubtitle(text) {
    let subtitleDiv = document.getElementById("asl-subtitle");
    if (!subtitleDiv) {
        subtitleDiv = document.createElement("div");
        subtitleDiv.id = "asl-subtitle";
        subtitleDiv.style.position = "absolute";
        subtitleDiv.style.bottom = "10px";
        subtitleDiv.style.left = "50%";
        subtitleDiv.style.transform = "translateX(-50%)";
        subtitleDiv.style.background = "rgba(0, 0, 0, 0.7)";
        subtitleDiv.style.color = "white";
        subtitleDiv.style.padding = "5px 10px";
        subtitleDiv.style.borderRadius = "5px";
        subtitleDiv.style.fontSize = "16px";
        subtitleDiv.style.zIndex = "9999";
        document.body.appendChild(subtitleDiv);
    }
    subtitleDiv.innerText = text;
}

// Function to attempt reconnection
function reconnectExtension() {
    setTimeout(() => {
        console.log("Re-injecting content script...");
        fetchSubtitle();  // Try to re-establish the connection
    }, 2000);  // Wait 2 seconds before retrying
}

// Try to fetch subtitles every 2 seconds
setInterval(fetchSubtitle, 2000);
