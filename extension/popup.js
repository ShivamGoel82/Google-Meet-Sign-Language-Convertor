document.addEventListener("DOMContentLoaded", () => {
    const toggleSwitch = document.getElementById("toggleSubtitles");
    const statusText = document.getElementById("status");

    // Load stored preference
    chrome.storage.sync.get("subtitlesEnabled", (data) => {
        toggleSwitch.checked = data.subtitlesEnabled || false;
        updateStatus();
    });

    toggleSwitch.addEventListener("change", () => {
        const enabled = toggleSwitch.checked;
        chrome.storage.sync.set({ subtitlesEnabled: enabled });
        updateStatus();

        // Notify content script to enable/disable subtitles
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, { type: "toggle_subtitles", enabled });
            }
        });
    });

    function updateStatus() {
        statusText.textContent = toggleSwitch.checked ? "Subtitles: On" : "Subtitles: Off";
    }
});
