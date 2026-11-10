const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const chatEndpoint = "/api/chat";
const uploadForm = document.querySelector("#upload-form");
const documentInput = document.querySelector("#document-file");
const uploadButton = document.querySelector("#upload-button");
const uploadStatus = document.querySelector("#upload-status");
let uploadInProgress = false;

// Add a message using textContent so chat text is treated as plain text.
function addMessage(text, sender, isError = false, sources = []) {
  const row = document.createElement("article");
  row.className = `message-row ${sender}-row${isError ? " error-row" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${sender}-avatar`;
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = sender === "assistant" ? "✦" : "You";

  const content = document.createElement("div");
  content.className = "message-content";

  const author = document.createElement("span");
  author.className = "message-author";
  author.textContent = sender === "assistant" ? "Gemini" : "You";

  const bubble = document.createElement("p");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  content.append(author, bubble);

  // Show the source filenames returned with a successful Gemini response.
  if (sender === "assistant" && !isError) {
    const sourceSection = document.createElement("section");
    sourceSection.className = "message-sources";
    sourceSection.setAttribute("aria-label", "Sources used");

    const sourceHeading = document.createElement("span");
    sourceHeading.className = "sources-heading";
    sourceHeading.textContent = "Sources used";
    sourceSection.append(sourceHeading);

    const uniqueSources = Array.isArray(sources)
      ? [...new Set(sources.filter((source) => typeof source === "string" && source.trim()))]
      : [];

    if (uniqueSources.length > 0) {
      const sourceList = document.createElement("ul");
      sourceList.className = "sources-list";

      uniqueSources.forEach((source) => {
        const sourceItem = document.createElement("li");
        sourceItem.textContent = source;
        sourceList.append(sourceItem);
      });

      sourceSection.append(sourceList);
    } else {
      const emptySources = document.createElement("p");
      emptySources.className = "no-sources";
      emptySources.textContent = "No relevant documents were retrieved.";
      sourceSection.append(emptySources);
    }

    content.append(sourceSection);
  }

  row.append(avatar, content);
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

// Show animated dots while waiting for the Flask API.
function addLoadingIndicator() {
  const row = document.createElement("article");
  row.className = "message-row assistant-row";
  row.setAttribute("role", "status");
  row.setAttribute("aria-label", "Gemini is generating a response");

  const avatar = document.createElement("div");
  avatar.className = "avatar assistant-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "✦";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble loading-bubble";
  bubble.setAttribute("aria-hidden", "true");

  for (let dot = 0; dot < 3; dot += 1) {
    bubble.append(document.createElement("span"));
  }

  row.append(avatar, bubble);
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = questionInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  questionInput.value = "";
  questionInput.disabled = true;
  sendButton.disabled = true;

  const loadingIndicator = addLoadingIndicator();

  try {
    const response = await fetch(chatEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The server returned an invalid response.");
    }
    if (!response.ok) {
      throw new Error(data.error || "The request could not be completed.");
    }

    if (typeof data.response !== "string" || !data.response.trim()) {
      throw new Error("The server returned an empty response.");
    }

    addMessage(data.response, "assistant", false, data.sources);
  } catch (error) {
    // Keep details useful to the user without exposing server-side secrets.
    const errorMessage = error instanceof TypeError
      ? "Could not connect to the chatbot. Check that Flask is running and try again."
      : error.message;
    addMessage(errorMessage, "assistant", true);
  } finally {
    loadingIndicator.remove();
    questionInput.disabled = false;
    sendButton.disabled = false;
    questionInput.focus();
  }
});

if (uploadForm && documentInput && uploadButton && uploadStatus) {
  // Show the selected filename before the user uploads it.
  documentInput.addEventListener("change", () => {
    const selectedFile = documentInput.files[0];
    uploadStatus.textContent = selectedFile
      ? `Selected: ${selectedFile.name}`
      : "Select a PDF or TXT document to upload.";
  });

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (uploadInProgress) return;

    const selectedFile = documentInput.files[0];
    if (!selectedFile) {
      uploadStatus.textContent = "Choose a PDF or TXT file before uploading.";
      return;
    }

    // Check the filename extension as well as the file input's accept filter.
    const extension = selectedFile.name.split(".").pop().toLowerCase();
    if (!["pdf", "txt"].includes(extension)) {
      uploadStatus.textContent = "Only PDF and TXT files can be uploaded.";
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    uploadInProgress = true;
    uploadButton.disabled = true;
    documentInput.disabled = true;
    uploadStatus.textContent = `Uploading ${selectedFile.name}...`;

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error("The server returned an invalid upload response.");
      }

      if (!response.ok || !data || data.success !== true) {
        throw new Error(data?.error || "The document could not be uploaded.");
      }

      const uploadedFilename = data.filename || selectedFile.name;
      const successMessage = data.message || `Uploaded ${uploadedFilename} successfully.`;
      uploadStatus.textContent = `${successMessage} File: ${uploadedFilename}`;
      documentInput.value = "";
    } catch (error) {
      uploadStatus.textContent = error instanceof TypeError
        ? "Could not connect to the server. Check that Flask is running and try again."
        : error.message;
    } finally {
      uploadInProgress = false;
      uploadButton.disabled = false;
      documentInput.disabled = false;
    }
  });
}
