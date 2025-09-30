/**
 * Background Service Worker for Phishing Shield Extension
 * Handles URL checking, API communication, and notification management
 */

// Configuration
const CONFIG = {
  API_URL: 'http://localhost:8000/api/v1',
  CHECK_TIMEOUT: 5000,
  CACHE_DURATION: 3600000, // 1 hour
  RISK_THRESHOLD: 60,
};

// Cache for checked URLs
const urlCache = new Map();

// Statistics
let stats = {
  urlsChecked: 0,
  phishingBlocked: 0,
  safeSites: 0,
};

/**
 * Initialize extension
 */
chrome.runtime.onInstalled.addListener(async () => {
  console.log('Phishing Shield extension installed');

  // Set default settings
  await chrome.storage.local.set({
    enabled: true,
    apiUrl: CONFIG.API_URL,
    riskThreshold: CONFIG.RISK_THRESHOLD,
    blockHighRisk: true,
    showNotifications: true,
    stats: stats,
  });

  // Show welcome notification
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icons/icon128.png',
    title: 'Phishing Shield Activated',
    message: 'You are now protected from phishing attacks!',
  });
});

/**
 * Listen for navigation events
 */
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId === 0) { // Main frame only
    const url = details.url;

    // Skip internal pages and extensions
    if (url.startsWith('chrome://') || url.startsWith('chrome-extension://')) {
      return;
    }

    // Check if protection is enabled
    const { enabled } = await chrome.storage.local.get('enabled');
    if (!enabled) {
      return;
    }

    // Check URL
    checkURL(url, details.tabId);
  }
});

/**
 * Listen for completed navigation
 */
chrome.webNavigation.onCompleted.addListener(async (details) => {
  if (details.frameId === 0) {
    const url = details.url;

    // Update badge with status
    updateBadge(details.tabId, url);
  }
});

/**
 * Check URL for phishing
 */
async function checkURL(url, tabId) {
  try {
    // Check cache first
    const cached = getCachedResult(url);
    if (cached) {
      console.log('Using cached result for:', url);
      handleResult(cached, url, tabId);
      return;
    }

    // Get settings
    const settings = await chrome.storage.local.get([
      'apiUrl',
      'riskThreshold',
      'blockHighRisk',
      'showNotifications',
    ]);

    const apiUrl = settings.apiUrl || CONFIG.API_URL;

    // Call API
    const response = await fetch(`${apiUrl}/analyze/url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
      signal: AbortSignal.timeout(CONFIG.CHECK_TIMEOUT),
    });

    if (!response.ok) {
      console.error('API error:', response.status);
      return;
    }

    const result = await response.json();

    // Cache result
    cacheResult(url, result);

    // Update statistics
    stats.urlsChecked++;
    if (result.is_phishing) {
      stats.phishingBlocked++;
    } else {
      stats.safeSites++;
    }
    await chrome.storage.local.set({ stats });

    // Handle result
    handleResult(result, url, tabId);

  } catch (error) {
    console.error('Error checking URL:', error);
    // Don't block on error - fail open
  }
}

/**
 * Handle analysis result
 */
async function handleResult(result, url, tabId) {
  const settings = await chrome.storage.local.get([
    'riskThreshold',
    'blockHighRisk',
    'showNotifications',
  ]);

  const riskThreshold = settings.riskThreshold || CONFIG.RISK_THRESHOLD;
  const isHighRisk = result.risk_score >= riskThreshold;

  // Update badge
  if (isHighRisk) {
    chrome.action.setBadgeText({ text: '⚠️', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#DC2626', tabId });
  } else if (result.risk_score >= 40) {
    chrome.action.setBadgeText({ text: '!', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#F59E0B', tabId });
  } else {
    chrome.action.setBadgeText({ text: '✓', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#10B981', tabId });
  }

  // Block high-risk sites if enabled
  if (settings.blockHighRisk && isHighRisk) {
    blockPage(tabId, url, result);
    return;
  }

  // Show notification for phishing
  if (settings.showNotifications && result.is_phishing) {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: '⚠️ Phishing Warning!',
      message: `This site may be dangerous. Risk Score: ${result.risk_score.toFixed(0)}/100`,
      priority: 2,
    });
  }

  // Inject warning banner for medium risk
  if (result.risk_score >= 40 && result.risk_score < riskThreshold) {
    injectWarningBanner(tabId, result);
  }
}

/**
 * Block dangerous page
 */
function blockPage(tabId, url, result) {
  const blockPageUrl = chrome.runtime.getURL('warning.html') +
    `?url=${encodeURIComponent(url)}&score=${result.risk_score.toFixed(0)}`;

  chrome.tabs.update(tabId, { url: blockPageUrl });

  // Show notification
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icons/icon128.png',
    title: '🛑 Phishing Site Blocked!',
    message: `Blocked dangerous website with risk score ${result.risk_score.toFixed(0)}/100`,
    priority: 2,
  });
}

/**
 * Inject warning banner
 */
function injectWarningBanner(tabId, result) {
  chrome.scripting.executeScript({
    target: { tabId },
    func: (riskScore, warnings) => {
      // Check if banner already exists
      if (document.getElementById('phishing-shield-banner')) {
        return;
      }

      // Create banner
      const banner = document.createElement('div');
      banner.id = 'phishing-shield-banner';
      banner.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #F59E0B 0%, #DC2626 100%);
        color: white;
        padding: 15px 20px;
        z-index: 999999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: slideDown 0.3s ease-out;
      `;

      banner.innerHTML = `
        <div style="display: flex; align-items: center; gap: 15px;">
          <span style="font-size: 24px;">⚠️</span>
          <div>
            <div style="font-weight: bold; font-size: 16px;">
              Phishing Warning - Risk Score: ${riskScore}/100
            </div>
            <div style="font-size: 13px; opacity: 0.9; margin-top: 4px;">
              This website may be attempting to steal your information. Exercise caution.
            </div>
          </div>
        </div>
        <button id="phishing-shield-close" style="
          background: rgba(255,255,255,0.2);
          border: 1px solid rgba(255,255,255,0.3);
          color: white;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
        ">
          Dismiss
        </button>
      `;

      // Add animation
      const style = document.createElement('style');
      style.textContent = `
        @keyframes slideDown {
          from { transform: translateY(-100%); }
          to { transform: translateY(0); }
        }
      `;
      document.head.appendChild(style);

      // Insert banner
      document.body.insertBefore(banner, document.body.firstChild);

      // Adjust body padding
      document.body.style.paddingTop = banner.offsetHeight + 'px';

      // Close button
      document.getElementById('phishing-shield-close').addEventListener('click', () => {
        banner.remove();
        document.body.style.paddingTop = '0';
      });
    },
    args: [result.risk_score.toFixed(0), result.warnings || []],
  });
}

/**
 * Update badge for tab
 */
async function updateBadge(tabId, url) {
  const cached = getCachedResult(url);
  if (cached) {
    handleResult(cached, url, tabId);
  }
}

/**
 * Cache management
 */
function cacheResult(url, result) {
  urlCache.set(url, {
    result,
    timestamp: Date.now(),
  });

  // Clean old cache entries
  const now = Date.now();
  for (const [key, value] of urlCache.entries()) {
    if (now - value.timestamp > CONFIG.CACHE_DURATION) {
      urlCache.delete(key);
    }
  }
}

function getCachedResult(url) {
  const cached = urlCache.get(url);
  if (cached && Date.now() - cached.timestamp < CONFIG.CACHE_DURATION) {
    return cached.result;
  }
  return null;
}

/**
 * Message handler
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'checkURL') {
    checkURL(message.url, sender.tab?.id).then(() => {
      sendResponse({ success: true });
    }).catch(error => {
      sendResponse({ success: false, error: error.message });
    });
    return true; // Keep message channel open
  }

  if (message.action === 'getStats') {
    chrome.storage.local.get('stats').then(data => {
      sendResponse(data.stats || stats);
    });
    return true;
  }

  if (message.action === 'getCurrentURL') {
    chrome.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      if (tabs[0]) {
        const url = tabs[0].url;
        const cached = getCachedResult(url);
        sendResponse({ url, result: cached });
      }
    });
    return true;
  }

  if (message.action === 'clearCache') {
    urlCache.clear();
    sendResponse({ success: true });
    return true;
  }
});

/**
 * Context menu
 */
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'checkURL',
    title: 'Check this link with Phishing Shield',
    contexts: ['link'],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'checkURL' && info.linkUrl) {
    checkURL(info.linkUrl, tab.id);
  }
});

console.log('Phishing Shield background service worker loaded');
