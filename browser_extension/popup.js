/**
 * Popup JavaScript for Phishing Shield Extension
 * Handles UI interactions and displays analysis results
 */

// DOM Elements
let statusBadge, statusIcon, statusText, urlDisplay, riskMeter, riskScore, riskMeterFill;
let infoGrid, classification, probability, warningsSection, warningsList;
let checkBtn, reportBtn, settingsBtn, protectionToggle;
let statChecked, statBlocked, statSafe;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  initializeElements();
  loadSettings();
  loadStats();
  checkCurrentPage();
  setupEventListeners();
});

/**
 * Initialize DOM elements
 */
function initializeElements() {
  statusBadge = document.getElementById('statusBadge');
  statusIcon = document.getElementById('statusIcon');
  statusText = document.getElementById('statusText');
  urlDisplay = document.getElementById('urlDisplay');
  riskMeter = document.getElementById('riskMeter');
  riskScore = document.getElementById('riskScore');
  riskMeterFill = document.getElementById('riskMeterFill');
  infoGrid = document.getElementById('infoGrid');
  classification = document.getElementById('classification');
  probability = document.getElementById('probability');
  warningsSection = document.getElementById('warningsSection');
  warningsList = document.getElementById('warningsList');
  checkBtn = document.getElementById('checkBtn');
  reportBtn = document.getElementById('reportBtn');
  settingsBtn = document.getElementById('settingsBtn');
  protectionToggle = document.getElementById('protectionToggle');
  statChecked = document.getElementById('statChecked');
  statBlocked = document.getElementById('statBlocked');
  statSafe = document.getElementById('statSafe');
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  checkBtn.addEventListener('click', handleCheckClick);
  reportBtn.addEventListener('click', handleReportClick);
  settingsBtn.addEventListener('click', handleSettingsClick);
  protectionToggle.addEventListener('change', handleProtectionToggle);
}

/**
 * Load settings
 */
async function loadSettings() {
  const settings = await chrome.storage.local.get(['enabled']);
  protectionToggle.checked = settings.enabled !== false;
}

/**
 * Load statistics
 */
async function loadStats() {
  const { stats } = await chrome.storage.local.get('stats');

  if (stats) {
    statChecked.textContent = stats.urlsChecked || 0;
    statBlocked.textContent = stats.phishingBlocked || 0;
    statSafe.textContent = stats.safeSites || 0;
  }
}

/**
 * Check current page
 */
async function checkCurrentPage() {
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.url) {
      showStatus('error', 'Unable to check this page');
      return;
    }

    const url = tab.url;

    // Skip internal pages
    if (url.startsWith('chrome://') || url.startsWith('chrome-extension://')) {
      showStatus('safe', 'Internal page - no check needed');
      urlDisplay.textContent = 'Chrome internal page';
      return;
    }

    // Display URL
    urlDisplay.textContent = url;

    // Request analysis from background
    chrome.runtime.sendMessage(
      { action: 'getCurrentURL' },
      (response) => {
        if (response && response.result) {
          displayResult(response.result, url);
        } else {
          showStatus('checking', 'Analyzing page...');
          // Trigger check
          chrome.runtime.sendMessage(
            { action: 'checkURL', url: url },
            () => {
              // Wait a bit for result
              setTimeout(() => {
                chrome.runtime.sendMessage(
                  { action: 'getCurrentURL' },
                  (resp) => {
                    if (resp && resp.result) {
                      displayResult(resp.result, url);
                    } else {
                      showStatus('safe', 'Analysis complete');
                    }
                  }
                );
              }, 2000);
            }
          );
        }
      }
    );
  } catch (error) {
    console.error('Error checking page:', error);
    showStatus('error', 'Check failed');
  }
}

/**
 * Display analysis result
 */
function displayResult(result, url) {
  // Update URL display
  urlDisplay.textContent = url;

  // Update status badge
  const riskScoreValue = Math.round(result.risk_score || 0);

  if (result.is_phishing || riskScoreValue >= 60) {
    showStatus('danger', 'Phishing Detected!');
    reportBtn.classList.remove('hidden');
  } else if (riskScoreValue >= 40) {
    showStatus('warning', 'Suspicious Site');
    reportBtn.classList.remove('hidden');
  } else {
    showStatus('safe', 'Site Appears Safe');
    reportBtn.classList.add('hidden');
  }

  // Show risk meter
  riskMeter.classList.remove('hidden');
  riskScore.textContent = `${riskScoreValue}/100`;
  riskMeterFill.style.width = `${riskScoreValue}%`;

  // Update risk meter color
  riskMeterFill.className = 'risk-meter-fill';
  if (riskScoreValue >= 80) {
    riskMeterFill.classList.add('risk-critical');
  } else if (riskScoreValue >= 60) {
    riskMeterFill.classList.add('risk-high');
  } else if (riskScoreValue >= 40) {
    riskMeterFill.classList.add('risk-medium');
  } else {
    riskMeterFill.classList.add('risk-low');
  }

  // Show info grid
  infoGrid.classList.remove('hidden');
  classification.textContent = result.classification || 'N/A';

  const probabilityValue = result.phishing_probability
    ? Math.round(result.phishing_probability * 100)
    : 0;
  probability.textContent = `${probabilityValue}%`;

  // Display warnings
  if (result.warnings && result.warnings.length > 0) {
    warningsSection.classList.add('active');
    warningsList.innerHTML = '';

    result.warnings.slice(0, 5).forEach(warning => {
      const warningItem = document.createElement('div');
      warningItem.className = 'warning-item';
      warningItem.textContent = warning;
      warningsList.appendChild(warningItem);
    });
  } else {
    warningsSection.classList.remove('active');
  }

  // Update statistics
  loadStats();
}

/**
 * Show status
 */
function showStatus(type, message) {
  statusBadge.className = `status-badge ${type}`;
  statusText.textContent = message;

  switch (type) {
    case 'safe':
      statusIcon.textContent = '✅';
      break;
    case 'warning':
      statusIcon.textContent = '⚠️';
      break;
    case 'danger':
      statusIcon.textContent = '🚨';
      break;
    case 'checking':
      statusIcon.textContent = '🔍';
      break;
    case 'error':
      statusIcon.textContent = '❌';
      break;
    default:
      statusIcon.textContent = '🛡️';
  }
}

/**
 * Handle check button click
 */
async function handleCheckClick() {
  checkBtn.disabled = true;
  checkBtn.innerHTML = '<span>🔄</span><span>Checking...</span>';

  await checkCurrentPage();

  setTimeout(() => {
    checkBtn.disabled = false;
    checkBtn.innerHTML = '<span>🔍</span><span>Check This Page</span>';
  }, 1000);
}

/**
 * Handle report button click
 */
async function handleReportClick() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (tab && tab.url) {
    // Open report form (could be a new tab or modal)
    const reportUrl = `https://safebrowsing.google.com/safebrowsing/report_phish/?url=${encodeURIComponent(tab.url)}`;
    chrome.tabs.create({ url: reportUrl });
  }
}

/**
 * Handle settings button click
 */
function handleSettingsClick() {
  chrome.runtime.openOptionsPage();
}

/**
 * Handle protection toggle
 */
async function handleProtectionToggle() {
  const enabled = protectionToggle.checked;
  await chrome.storage.local.set({ enabled });

  // Show notification
  if (enabled) {
    showStatus('safe', 'Protection Enabled');
  } else {
    showStatus('warning', 'Protection Disabled');
  }

  // Refresh check
  setTimeout(checkCurrentPage, 500);
}

/**
 * Format number with commas
 */
function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Truncate URL for display
 */
function truncateUrl(url, maxLength = 50) {
  if (url.length <= maxLength) {
    return url;
  }
  return url.substring(0, maxLength - 3) + '...';
}

// Refresh stats periodically
setInterval(loadStats, 5000);
