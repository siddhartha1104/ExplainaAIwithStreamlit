document.addEventListener("DOMContentLoaded", function () {
  // Theme toggle functionality
  const themeToggle = document.getElementById("theme-toggle");
  const htmlElement = document.documentElement;

  // Check for saved theme preference or use system preference
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme) {
    htmlElement.classList.remove("dark", "light");
    htmlElement.classList.add(savedTheme);
  } else if (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    htmlElement.classList.add("dark");
    htmlElement.classList.remove("light");
  }

  // Update icons based on current theme
  updateThemeIcons();

  themeToggle.addEventListener("click", function () {
    if (htmlElement.classList.contains("dark")) {
      htmlElement.classList.remove("dark");
      htmlElement.classList.add("light");
      localStorage.setItem("theme", "light");
    } else {
      htmlElement.classList.remove("light");
      htmlElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    }
    updateThemeIcons();
  });

  function updateThemeIcons() {
    const moonIcon = themeToggle.querySelector(".fa-moon");
    const sunIcon = themeToggle.querySelector(".fa-sun");

    if (htmlElement.classList.contains("dark")) {
      moonIcon.classList.add("hidden");
      sunIcon.classList.remove("hidden");
    } else {
      moonIcon.classList.remove("hidden");
      sunIcon.classList.add("hidden");
    }
  }

  // Auto-resize textarea
  const textarea = document.getElementById("questionInput");
  textarea.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height =
      (this.scrollHeight < 150 ? this.scrollHeight : 150) + "px";
  });

  // URL Form Submission
  const urlForm = document.getElementById("urlForm");
  const urlInput = document.getElementById("urlInput");
  const processButtonText = document.getElementById("processButtonText");
  const loadingIndicator = document.getElementById("loadingIndicator");

  urlForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const url = urlInput.value.trim();

    if (!url) {
      showNotification("Please enter a URL", "error");
      return;
    }

    // Show loading indicator
    processButtonText.classList.add("hidden");
    loadingIndicator.classList.remove("hidden");

    // Prepare form data
    const formData = new FormData();
    formData.append("url", url);

    // Send request to server
    fetch("/process_url", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Hide loading indicator
        processButtonText.classList.remove("hidden");
        loadingIndicator.classList.add("hidden");

        if (data.success) {
          handleSuccessfulUrlProcessing(data, url);
        } else {
          showNotification(data.error, "error");
        }
      })
      .catch((error) => {
        processButtonText.classList.remove("hidden");
        loadingIndicator.classList.add("hidden");
        showNotification("An error occurred: " + error, "error");
      });
  });

  function handleSuccessfulUrlProcessing(data, url) {
    // Update UI to show content info
    document.getElementById("noContentSection").classList.add("hidden");
    document.getElementById("contentInfoDetails").classList.remove("hidden");
    document.getElementById("contentSource").textContent = data.content_source;

    // Show corresponding content type info section
    const youtubeInfoSection = document.getElementById("youtubeInfoSection");
    const wikipediaInfoSection = document.getElementById(
      "wikipediaInfoSection"
    );
    const webpageInfoSection = document.getElementById("webpageInfoSection");

    youtubeInfoSection.classList.add("hidden");
    wikipediaInfoSection.classList.add("hidden");
    webpageInfoSection.classList.add("hidden");

    if (data.url_type === "youtube") {
      youtubeInfoSection.classList.remove("hidden");
      // Set YouTube thumbnail
      const videoId = data.media_info;
      document.getElementById(
        "youtubeThumbnail"
      ).style.backgroundImage = `url(https://img.youtube.com/vi/${videoId}/mqdefault.jpg)`;
    } else if (data.url_type === "wikipedia") {
      wikipediaInfoSection.classList.remove("hidden");
    } else {
      webpageInfoSection.classList.remove("hidden");

      // Set favicon
      if (data.media_info) {
        document.getElementById("faviconImg").src = data.media_info;
      } else {
        document.getElementById("faviconImg").src =
          "https://www.google.com/s2/favicons?domain=" +
          new URL(url).hostname +
          "&sz=64";
      }
    }

    // Hide landing page, show chat interface
    document.getElementById("landingPage").classList.add("hidden");
    document.getElementById("chatInterface").classList.remove("hidden");

    // Display initial chat message
    displayChatHistory(data.chat_history);

    // Show success notification
    showNotification("Content processed successfully!", "success");
  }

  // Chat Form Submission
  const chatForm = document.getElementById("chatForm");

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const question = document.getElementById("questionInput").value.trim();

    if (!question) return;

    // Clear the input field
    document.getElementById("questionInput").value = "";
    document.getElementById("questionInput").style.height = "auto";

    // Add user message to chat
    addMessageToChat("user", question);

    // Show typing indicator
    addTypingIndicator();

    // Prepare form data
    const formData = new FormData();
    formData.append("question", question);

    // Send request to server
    fetch("/chat", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Remove typing indicator
        removeTypingIndicator();

        if (data.success) {
          // Add assistant message to chat
          addMessageToChat("assistant", data.answer);
        } else {
          showNotification(data.error, "error");
        }
      })
      .catch((error) => {
        removeTypingIndicator();
        showNotification("An error occurred: " + error, "error");
      });
  });

  // View Content Buttons
  document
    .getElementById("viewSummaryBtn")
    .addEventListener("click", function () {
      fetch("/get_content?type=summary")
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            showContentModal("Content Summary", data.content);
          } else {
            showNotification(data.error, "error");
          }
        })
        .catch((error) =>
          showNotification("An error occurred: " + error, "error")
        );
    });

  document
    .getElementById("viewFullContentBtn")
    .addEventListener("click", function () {
      fetch("/get_content?type=extracted")
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            showContentModal("Full Content", data.content);
          } else {
            showNotification(data.error, "error");
          }
        })
        .catch((error) =>
          showNotification("An error occurred: " + error, "error")
        );
    });

  // Clear Conversation Button
  document
    .getElementById("clearConversationBtn")
    .addEventListener("click", function () {
      if (confirm("Are you sure you want to clear the conversation history?")) {
        fetch("/clear_conversation", { method: "POST" })
          .then((response) => response.json())
          .then((data) => {
            if (data.success) {
              // Update chat with just the initial message
              displayChatHistory(data.chat_history);
              showNotification("Conversation cleared", "success");
            } else {
              showNotification(data.error, "error");
            }
          })
          .catch((error) =>
            showNotification("An error occurred: " + error, "error")
          );
      }
    });

  // Modal Close Button
  document
    .getElementById("closeModalBtn")
    .addEventListener("click", function () {
      document.getElementById("contentModal").classList.add("hidden");
    });

  // Close modal when clicking outside
  const contentModal = document.getElementById("contentModal");
  contentModal.addEventListener("click", function (e) {
    if (e.target === contentModal) {
      contentModal.classList.add("hidden");
    }
  });

  // Notification system
  function showNotification(message, type) {
    // Create notification element if it doesn't exist
    let notification = document.getElementById("notification");
    if (!notification) {
      notification = document.createElement("div");
      notification.id = "notification";
      notification.className =
        "fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transform transition-transform duration-300 translate-x-full";
      document.body.appendChild(notification);
    }

    // Set notification type
    if (type === "success") {
      notification.className =
        "fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transform transition-transform duration-300 bg-green-500 text-white";
    } else if (type === "error") {
      notification.className =
        "fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transform transition-transform duration-300 bg-red-500 text-white";
    } else {
      notification.className =
        "fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transform transition-transform duration-300 bg-blue-500 text-white";
    }

    // Set message
    notification.textContent = message;

    // Show notification
    setTimeout(() => {
      notification.style.transform = "translateX(0)";
    }, 100);

    // Hide notification after 3 seconds
    setTimeout(() => {
      notification.style.transform = "translateX(100%)";
    }, 3000);
  }

  // Display chat history
  function displayChatHistory(history) {
    const chatMessages = document.getElementById("chatMessages");
    chatMessages.innerHTML = "";

    history.forEach((message) => {
      addMessageToChat(message.role, message.content, false);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Add a message to the chat
  function addMessageToChat(role, content, scroll = true) {
    const chatMessages = document.getElementById("chatMessages");
    const messageDiv = document.createElement("div");
    messageDiv.className =
      role === "user"
        ? "user-message self-end p-4 mb-4"
        : "assistant-message self-start p-4 mb-4";

    // Process content to add markdown formatting
    const formattedContent = formatMarkdown(content);
    messageDiv.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0 mr-3">
                    <div class="w-8 h-8 rounded-full flex items-center justify-center ${
                      role === "user"
                        ? "bg-accent text-white"
                        : "bg-gray-200 dark:bg-gray-700"
                    }">
                        <i class="${
                          role === "user" ? "fas fa-user" : "fas fa-robot"
                        }"></i>
                    </div>
                </div>
                <div class="markdown">${formattedContent}</div>
            </div>
        `;

    chatMessages.appendChild(messageDiv);

    if (scroll) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  // Add typing indicator
  function addTypingIndicator() {
    const chatMessages = document.getElementById("chatMessages");
    const typingDiv = document.createElement("div");
    typingDiv.id = "typingIndicator";
    typingDiv.className = "assistant-message self-start p-4";
    typingDiv.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0 mr-3">
                    <div class="w-8 h-8 rounded-full flex items-center justify-center bg-gray-200 dark:bg-gray-700">
                        <i class="fas fa-robot"></i>
                    </div>
                </div>
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Remove typing indicator
  function removeTypingIndicator() {
    const typingIndicator = document.getElementById("typingIndicator");
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  // Show content modal
  function showContentModal(title, content) {
    document.getElementById("modalTitle").textContent = title;
    document.getElementById("modalContent").innerHTML = formatMarkdown(content);
    document.getElementById("contentModal").classList.remove("hidden");
  }

  // Simple markdown formatting
  function formatMarkdown(text) {
    // Handle code blocks
    text = text.replace(/```([\s\S]*?)```/g, function (match, code) {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Handle inline code
    text = text.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Handle headers
    text = text.replace(/^### (.*$)/gm, "<h3>$1</h3>");
    text = text.replace(/^## (.*$)/gm, "<h2>$1</h2>");
    text = text.replace(/^# (.*$)/gm, "<h1>$1</h1>");

    // Handle bold
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Handle italic
    text = text.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // Handle lists
    text = text.replace(/^\s*[\-\*]\s+(.*$)/gm, "<ul><li>$1</li></ul>");
    text = text.replace(/^\s*\d+\.\s+(.*$)/gm, "<ol><li>$1</li></ol>");

    // Handle paragraphs
    text = text.replace(/^(?!<[houbl])(.+)$/gm, function (match) {
      return match.trim() ? `<p>${match}</p>` : "";
    });

    // Fix consecutive list items
    text = text.replace(/<\/ul>\s*<ul>/g, "");
    text = text.replace(/<\/ol>\s*<ol>/g, "");

    // Handle blockquotes
    text = text.replace(/^\> (.*$)/gm, "<blockquote>$1</blockquote>");

    // Handle line breaks
    text = text.replace(/\n\n/g, "<br><br>");

    return text;
  }

  // Check URL parameters on load
  const urlParams = new URLSearchParams(window.location.search);
  const autoUrl = urlParams.get("url");

  if (autoUrl) {
    urlInput.value = autoUrl;
    urlForm.dispatchEvent(new Event("submit"));
  }

  // Keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    // Ctrl/Cmd + Enter to submit chat
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      if (document.activeElement === textarea) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });
});
