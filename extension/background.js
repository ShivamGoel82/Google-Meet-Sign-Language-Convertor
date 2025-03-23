chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "sendVideoFrame") {
        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: JSON.stringify({ image: request.image }),
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then(response => response.json())
        .then(data => sendResponse({ subtitle: data.text }))
        .catch(error => console.error("Error:", error));
        return true; // Keeps the message channel open for async response
    }
});
