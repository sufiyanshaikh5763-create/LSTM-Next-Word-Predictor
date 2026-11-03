const promptInput = document.querySelector("#prompt");
const predictButton = document.querySelector("#predict-button");
const clearButton = document.querySelector("#clear-button");
const results = document.querySelector("#results");
const wordSlider = document.querySelector("#word-slider");
const wordValue = document.querySelector("#word-value");
let selectedWords = Number(wordSlider.value);

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2600);
}

promptInput.addEventListener("input", () => {
  document.querySelector("#character-count").textContent = `${promptInput.value.length} / 2000`;
});

function updateSlider() {
  selectedWords = Number(wordSlider.value);
  wordValue.textContent = selectedWords;
  wordSlider.style.setProperty("--slider-progress", `${((selectedWords - 1) / 9) * 100}%`);
}

wordSlider.addEventListener("input", updateSlider);
updateSlider();

clearButton.addEventListener("click", () => {
  promptInput.value = "";
  document.querySelector("#character-count").textContent = "0 / 2000";
  results.classList.add("hidden");
  promptInput.focus();
});

predictButton.addEventListener("click", async () => {
  const text = promptInput.value.trim();
  if (!text) {
    showToast("Enter a sentence to begin.");
    promptInput.focus();
    return;
  }
  predictButton.classList.add("loading");
  predictButton.disabled = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, num_words: selectedWords}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    document.querySelector("#original-output").textContent = data.original_input;
    document.querySelector("#continuation-output").textContent = data.predicted_continuation;
    document.querySelector("#sentence-output").textContent = data.generated_sentence;
    document.querySelector("#word-count").textContent = data.number_of_words;
    document.querySelector("#timing").textContent = `${data.execution_time_ms} ms`;
    results.classList.remove("hidden");
    results.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    showToast(error.message);
  } finally {
    predictButton.classList.remove("loading");
    predictButton.disabled = false;
  }
});

document.querySelector("#copy-button").addEventListener("click", async () => {
  const sentence = document.querySelector("#sentence-output").textContent;
  try {
    await navigator.clipboard.writeText(sentence);
    showToast("Generated text copied.");
  } catch {
    showToast("Could not copy text.");
  }
});

fetch("/api/health")
  .then((response) => response.json())
  .then((data) => {
    document.querySelector("#model-status").textContent = data.model_loaded ? "Model loaded" : "Model unavailable";
  })
  .catch(() => {
    document.querySelector("#model-status").textContent = "Backend offline";
  });
