document.getElementById("toggle").addEventListener("click", () => {
    chrome.storage.local.get(["enabled"], (result) => {
        let newState = !result.enabled;
        chrome.storage.local.set({ enabled: newState });

        document.getElementById("toggle").textContent = newState ? "Disable Subtitles" : "Enable Subtitles";
    });
});
