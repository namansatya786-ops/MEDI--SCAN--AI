const form = document.querySelector("#intake-form");
const controls = [...form.querySelectorAll(".form-control")];
const attachmentInput = document.querySelector("#attachment");
const attachmentError = document.querySelector("#attachment-error");
const dropzone = document.querySelector("#dropzone");
const uploadLabel = document.querySelector(".upload-label");
const filePreview = document.querySelector("#file-preview");
const imagePreview = document.querySelector("#image-preview");
const fileChip = document.querySelector("#file-chip");
const selectedFileName = document.querySelector("#selected-file-name");
const removeFileButton = document.querySelector("#remove-file");
const formStatus = document.querySelector("#form-status");
const results = document.querySelector("#results");
const predictionList = document.querySelector("#prediction-list");
const resultNotice = document.querySelector(".result-notice");
const submitButton = document.querySelector("#submit-button");
const cancelButton = document.querySelector("#cancel-button");

document.querySelectorAll(".field-error").forEach((error) => {
  error.setAttribute("aria-live", "polite");
});

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const allowedExtensions = new Set(["jpg", "jpeg", "png", "webp", "pdf"]);
const imageExtensions = new Set(["jpg", "jpeg", "png", "webp"]);
const allowedMimeTypes = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
  pdf: "application/pdf",
};
const touchedFields = new Set();
let selectedFile = null;
let previewUrl = null;
let isSubmitting = false;

function getErrorElement(field) {
  return document.getElementById(`${field.id}-error`);
}

function setFieldError(field, error) {
  const errorElement = getErrorElement(field);
  field.setAttribute("aria-invalid", error ? "true" : "false");
  errorElement.textContent = error;
}

function validateField(field) {
  const value = field.value.trim();
  let error = "";

  if (field.required && !value) {
    error = "This field is required.";
  } else if (value && field.id === "patient-age") {
    const age = Number(value);
    if (!Number.isInteger(age) || age < 1 || age > 120) {
      error = "Enter a whole number from 1 to 120.";
    }
  } else if (value && field.id === "patient-email") {
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(value)) {
      error = "Enter a valid email address.";
    }
  } else if (value && field.id === "patient-phone") {
    const digits = value.replace(/\D/g, "").length;
    const parenthesesAreBalanced = (value.match(/\(/g) || []).length === (value.match(/\)/g) || []).length;
    if (!/^\+?[\d().\-\s]+$/.test(value) || !parenthesesAreBalanced || digits < 7 || digits > 15) {
      error = "Enter a valid phone number with 7–15 digits.";
    }
  } else if (value && field.id === "blood-pressure") {
    const match = value.match(/^(\d{2,3})\s*\/\s*(\d{2,3})$/);
    if (!match || Number(match[1]) < 40 || Number(match[1]) > 300 ||
        Number(match[2]) < 20 || Number(match[2]) > 200) {
      error = "Use systolic/diastolic format, for example 120/80.";
    }
  } else if (value && field.id === "temperature") {
    const temperature = Number(value);
    if (!Number.isFinite(temperature) || temperature < 30 || temperature > 45) {
      error = "Enter a temperature from 30 to 45 °C.";
    } else if (field.validity.stepMismatch) {
      error = "Enter no more than one decimal place.";
    }
  }

  setFieldError(field, error);
  return !error;
}

function getFileExtension(file) {
  return file.name.split(".").pop().toLowerCase();
}

function validateAttachment(file) {
  if (!file) {
    attachmentInput.removeAttribute("aria-invalid");
    attachmentError.textContent = "";
    return true;
  }

  const extension = getFileExtension(file);
  let error = "";
  if (!allowedExtensions.has(extension)) {
    error = "Choose a JPG, PNG, WebP, or PDF file.";
  } else if (file.size === 0) {
    error = "The selected file is empty.";
  } else if (file.size > MAX_FILE_BYTES) {
    error = "The selected file must be 10 MB or smaller.";
  } else if (file.type && file.type !== allowedMimeTypes[extension]) {
    error = "The selected file type does not match its extension.";
  }

  attachmentInput.setAttribute("aria-invalid", error ? "true" : "false");
  attachmentError.textContent = error;
  return !error;
}

function revokePreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function setSelectedFile(file) {
  revokePreview();
  selectedFile = file;
  attachmentInput.value = "";
  validateAttachment(file);
  formStatus.hidden = true;
  results.hidden = true;

  if (!file) {
    filePreview.hidden = true;
    uploadLabel.hidden = false;
    imagePreview.hidden = true;
    imagePreview.removeAttribute("src");
    return;
  }

  selectedFileName.textContent = file.name;
  filePreview.hidden = false;
  uploadLabel.hidden = true;

  if (imageExtensions.has(getFileExtension(file))) {
    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    imagePreview.hidden = false;
    fileChip.hidden = true;
  } else {
    imagePreview.hidden = true;
    imagePreview.removeAttribute("src");
    fileChip.hidden = false;
    fileChip.textContent = getFileExtension(file) === "pdf" ? "PDF" : "FILE";
  }
}

function setStatus(message, tone = "success") {
  formStatus.textContent = message;
  formStatus.className = `form-status${tone === "success" ? "" : ` status-${tone}`}`;
  formStatus.hidden = false;
}

function clearResults() {
  predictionList.replaceChildren();
  resultNotice.textContent = "";
  results.hidden = true;
}

function renderPredictions(predictions, notice) {
  predictionList.replaceChildren();
  for (const prediction of predictions) {
    const item = document.createElement("li");
    item.className = "prediction";

    const label = document.createElement("span");
    label.textContent = prediction.class;

    const track = document.createElement("span");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.className = "score-fill";
    fill.style.display = "block";
    fill.style.width = `${Math.max(0, Math.min(100, prediction.score * 100))}%`;
    track.append(fill);

    const score = document.createElement("span");
    score.className = "score-value";
    score.textContent = `${(prediction.score * 100).toFixed(1)}%`;

    item.append(label, track, score);
    predictionList.append(item);
  }

  resultNotice.textContent = notice;
  results.hidden = false;
}

async function analyzeImage(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  const response = await fetch("/api/predict", { method: "POST", body });
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "The local model could not analyze this image.");
  }
  if (!Array.isArray(data.predictions) || typeof data.notice !== "string") {
    throw new Error("The local model returned an invalid response.");
  }

  renderPredictions(data.predictions, data.notice);
}

function resetFormState() {
  revokePreview();
  selectedFile = null;
  touchedFields.clear();
  controls.forEach((field) => {
    field.removeAttribute("aria-invalid");
    getErrorElement(field).textContent = "";
  });
  attachmentInput.removeAttribute("aria-invalid");
  attachmentError.textContent = "";
  filePreview.hidden = true;
  uploadLabel.hidden = false;
  imagePreview.hidden = true;
  imagePreview.removeAttribute("src");
  fileChip.hidden = false;
  formStatus.hidden = true;
  clearResults();
}

controls.forEach((field) => {
  field.addEventListener("blur", () => {
    touchedFields.add(field.id);
    validateField(field);
  });
  field.addEventListener("input", () => {
    if (touchedFields.has(field.id)) validateField(field);
    formStatus.hidden = true;
    clearResults();
  });
  field.addEventListener("change", () => {
    if (touchedFields.has(field.id)) validateField(field);
    formStatus.hidden = true;
    clearResults();
  });
});

attachmentInput.addEventListener("change", () => {
  setSelectedFile(attachmentInput.files[0] || null);
});

removeFileButton.addEventListener("click", () => setSelectedFile(null));

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
  });
}

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) setSelectedFile(file);
});

form.addEventListener("reset", () => {
  if (isSubmitting) return;
  window.setTimeout(resetFormState, 0);
});

cancelButton.addEventListener("click", () => {
  if (isSubmitting) return;
  form.reset();
  window.setTimeout(() => {
    setStatus("Form cleared. No information was submitted or saved.");
  }, 0);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isSubmitting) return;

  const fieldsAreValid = controls.map(validateField).every(Boolean);
  const fileIsValid = validateAttachment(selectedFile);
  if (!fieldsAreValid || !fileIsValid) {
    const firstInvalid = form.querySelector('[aria-invalid="true"]');
    firstInvalid?.focus();
    return;
  }

  isSubmitting = true;
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Validating…";
  cancelButton.disabled = true;
  form.querySelector('button[type="reset"]').disabled = true;
  form.setAttribute("aria-busy", "true");
  formStatus.hidden = true;
  clearResults();

  try {
    if (!selectedFile) {
      setStatus("Form submitted and validated in this browser. No patient details were sent or saved.");
    } else if (getFileExtension(selectedFile) === "pdf") {
      setStatus("Form submitted and validated. The PDF remains on this device and is not sent, saved, or analyzed.");
    } else {
      try {
        await analyzeImage(selectedFile);
        setStatus("Form submitted. The image was analyzed by the local model; patient details were not sent or saved.");
      } catch (error) {
        setStatus(
          `Form submitted, but image analysis did not complete: ${error.message} Only the image was sent to the local API; patient details were not sent.`,
          "warning",
        );
      }
    }
  } finally {
    isSubmitting = false;
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Submit";
    cancelButton.disabled = false;
    form.querySelector('button[type="reset"]').disabled = false;
    form.removeAttribute("aria-busy");
  }
});
