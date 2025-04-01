console.log("Background script running...");

const BACKEND_URL = "http://127.0.0.1:5000/predict";

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "fetch_subtitle") {
        fetch(BACKEND_URL)
            .then(response => response.json())
            .then(data => {
                if (data.text) {
                    chrome.tabs.sendMessage(sender.tab.id, { type: "update_subtitle", text: data.text });
                }
            })
            .catch(error => console.error("Error fetching subtitle:", error));
    }
});
