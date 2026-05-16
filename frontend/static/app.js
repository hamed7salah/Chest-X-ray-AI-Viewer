const uploadForm = document.getElementById('upload-form');
const imageInput = document.getElementById('image-file');
const statusEl = document.getElementById('status');
const predictionText = document.getElementById('prediction-text');
const scoreSummary = document.getElementById('score-summary');
const currentImage = document.getElementById('current-image');
const heatmapImage = document.getElementById('heatmap-image');
const priorSelect = document.getElementById('prior-select');
const feedbackSelect = document.getElementById('feedback-select');
const feedbackComment = document.getElementById('feedback-comment');
const submitFeedback = document.getElementById('submit-feedback');
const windowCenter = document.getElementById('window-center');
const windowWidth = document.getElementById('window-width');
const windowCenterValue = document.getElementById('window-center-value');
const windowWidthValue = document.getElementById('window-width-value');

let currentStudy = null;
let priorStudies = [];

function setStatus(message, type = 'info') {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

async function fetchStudies() {
  try {
    const response = await fetch('/api/studies');
    if (!response.ok) throw new Error('Unable to fetch prior studies');
    priorStudies = await response.json();
    priorSelect.innerHTML = '<option value="none">None</option>';
    priorStudies.forEach((study) => {
      const option = document.createElement('option');
      option.value = study.id;
      option.textContent = `${study.id}: ${study.patient_id ?? 'unknown'} (${study.prediction})`;
      priorSelect.appendChild(option);
    });
  } catch (error) {
    setStatus('Could not load prior studies.', 'error');
  }
}

function updateScoreSummary(study) {
  scoreSummary.innerHTML = `
    <div><strong>Score</strong>: ${study.score.toFixed(2)}</div>
    <div><strong>Ensemble</strong>: ${study.ensemble_score.toFixed(2)}</div>
    <div><strong>Uncertainty</strong>: ${study.uncertainty.toFixed(2)}</div>
  `;
}

function displayStudy(study) {
  currentStudy = study;
  predictionText.textContent = `${study.prediction} (score: ${study.score.toFixed(2)})`;
  updateScoreSummary(study);
  currentImage.src = `/backend-static${study.png_url}`;
  currentImage.style.filter = `contrast(${windowWidth.value}%) brightness(${windowCenter.value}%)`;
  if (study.heatmap_url) {
    heatmapImage.src = `/backend-static${study.heatmap_url}`;
    heatmapImage.style.display = 'block';
  } else {
    heatmapImage.style.display = 'none';
  }
  setStatus('Upload completed successfully.', 'success');
}

async function submitFile(event) {
  event.preventDefault();
  if (!imageInput.files.length) {
    setStatus('Select a file before uploading.', 'error');
    return;
  }
  const formData = new FormData();
  formData.append('file', imageInput.files[0]);
  setStatus('Uploading image...', 'info');

  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Upload failed');
    }
    const data = await response.json();
    displayStudy(data);
    await fetchStudies();
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

async function submitFeedbackHandler() {
  if (!currentStudy) {
    setStatus('Upload an image before submitting feedback.', 'error');
    return;
  }
  const reviewLabel = feedbackSelect.value;
  const comment = feedbackComment.value;
  if (reviewLabel === 'No change') {
    setStatus('Select a label change before submitting feedback.', 'warning');
    return;
  }

  try {
    const response = await fetch(`/api/studies/${currentStudy.id}/feedback`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({review_label: reviewLabel, comment}),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Feedback failed');
    }
    setStatus('Feedback submitted successfully.', 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

function updateWindowValues() {
  windowCenterValue.textContent = windowCenter.value;
  windowWidthValue.textContent = windowWidth.value;
  if (currentStudy) {
    currentImage.style.filter = `contrast(${windowWidth.value}%) brightness(${windowCenter.value}%)`;
    if (heatmapImage.src) heatmapImage.style.filter = `contrast(${windowWidth.value}%) brightness(${windowCenter.value}%)`;
  }
}

async function loadPriorStudy() {
  const selectedId = priorSelect.value;
  if (selectedId === 'none') return;
  const study = priorStudies.find((item) => String(item.id) === selectedId);
  if (!study) return;
  predictionText.textContent = `Prior: ${study.prediction} (score: ${study.score.toFixed(2)})`;
  updateScoreSummary(study);
  currentImage.src = `/backend-static${study.png_url}`;
  heatmapImage.src = study.heatmap_url ? `/backend-static${study.heatmap_url}` : '';
  heatmapImage.style.display = study.heatmap_url ? 'block' : 'none';
}

uploadForm.addEventListener('submit', submitFile);
submitFeedback.addEventListener('click', submitFeedbackHandler);
windowCenter.addEventListener('input', updateWindowValues);
windowWidth.addEventListener('input', updateWindowValues);
priorSelect.addEventListener('change', loadPriorStudy);

fetchStudies();
setStatus('Ready', 'info');
