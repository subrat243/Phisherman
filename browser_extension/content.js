/**
 * Content Script for Phishing Shield Extension
 * Runs on all web pages to provide real-time protection
 */

// Configuration
const SUSPICIOUS_KEYWORDS = [
  'verify',
  'confirm',
  'update',
  'suspend',
  'urgent',
  'security',
  'account',
  'login',
  'password',
  'click here',
];

// Track if we've already checked this page
let pageChecked = false;

/**
 * Initialize content script
 */
function init() {
  console.log('Phishing Shield: Content script loaded');

  // Check current URL
  checkCurrentPage();

  // Monitor form submissions
  monitorForms();

  // Monitor link clicks
  monitorLinks();

  // Check for suspicious patterns
  checkPageContent();
}

/**
 * Check current page URL
 */
function checkCurrentPage() {
  if (pageChecked) return;

  const url = window.location.href;

  // Send message to background script for analysis
  chrome.runtime.sendMessage(
    { action: 'checkURL', url: url },
    (response) => {
      if (response && response.success) {
        pageChecked = true;
      }
    }
  );
}

/**
 * Monitor form submissions
 */
function monitorForms() {
  const forms = document.querySelectorAll('form');

  forms.forEach((form) => {
    // Check for password fields
    const hasPasswordField = form.querySelector('input[type="password"]');

    if (hasPasswordField) {
      form.addEventListener('submit', (e) => {
        // Check if this is a suspicious site
        const url = window.location.href;

        // If not HTTPS, warn user
        if (!url.startsWith('https://')) {
          const confirmSubmit = confirm(
            '⚠️ WARNING: You are about to submit a form with a password on a non-secure (HTTP) connection.\n\n' +
            'Your password could be intercepted by attackers.\n\n' +
            'Do you want to continue?'
          );

          if (!confirmSubmit) {
            e.preventDefault();
            showWarningNotification('Form submission blocked for your safety');
          }
        }
      });
    }
  });
}

/**
 * Monitor link clicks
 */
function monitorLinks() {
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a');

    if (link && link.href) {
      const href = link.href;

      // Check for suspicious patterns
      if (isSuspiciousLink(href)) {
        e.preventDefault();

        const confirmNavigation = confirm(
          '⚠️ WARNING: This link appears suspicious!\n\n' +
          `URL: ${href}\n\n` +
          'Potential threats detected:\n' +
          getSuspiciousReasons(href) +
          '\n\nDo you want to continue?'
        );

        if (confirmNavigation) {
          window.location.href = href;
        }
      }
    }
  });
}

/**
 * Check if link is suspicious
 */
function isSuspiciousLink(url) {
  try {
    const urlObj = new URL(url);

    // Check for IP address
    const ipPattern = /^https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/;
    if (ipPattern.test(url)) {
      return true;
    }

    // Check for suspicious TLDs
    const suspiciousTLDs = ['tk', 'ml', 'ga', 'cf', 'gq', 'zip'];
    const hostname = urlObj.hostname;
    if (suspiciousTLDs.some(tld => hostname.endsWith('.' + tld))) {
      return true;
    }

    // Check for mismatched protocol (current page is HTTPS, link is HTTP)
    if (window.location.protocol === 'https:' && urlObj.protocol === 'http:') {
      return true;
    }

    // Check for unusual characters
    if (/[^\x00-\x7F]/.test(hostname)) {
      return true; // Contains non-ASCII characters (possible IDN homograph attack)
    }

    return false;
  } catch (e) {
    return false;
  }
}

/**
 * Get reasons why a link is suspicious
 */
function getSuspiciousReasons(url) {
  const reasons = [];

  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname;

    if (/^https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/.test(url)) {
      reasons.push('• Uses IP address instead of domain name');
    }

    const suspiciousTLDs = ['tk', 'ml', 'ga', 'cf', 'gq', 'zip'];
    if (suspiciousTLDs.some(tld => hostname.endsWith('.' + tld))) {
      reasons.push('• Uses suspicious domain extension (.tk, .ml, etc.)');
    }

    if (window.location.protocol === 'https:' && urlObj.protocol === 'http:') {
      reasons.push('• Not using secure HTTPS connection');
    }

    if (/[^\x00-\x7F]/.test(hostname)) {
      reasons.push('• Contains unusual characters (possible fake domain)');
    }

    if (url.length > 150) {
      reasons.push('• Unusually long URL');
    }

  } catch (e) {
    reasons.push('• Malformed URL');
  }

  return reasons.join('\n');
}

/**
 * Check page content for suspicious patterns
 */
function checkPageContent() {
  // Check for suspicious keywords in page content
  const pageText = document.body.innerText.toLowerCase();
  const suspiciousCount = SUSPICIOUS_KEYWORDS.filter(keyword =>
    pageText.includes(keyword)
  ).length;

  // Check for forms asking for sensitive information
  const passwordFields = document.querySelectorAll('input[type="password"]');
  const emailFields = document.querySelectorAll('input[type="email"]');
  const textFields = document.querySelectorAll('input[type="text"]');

  // Look for credit card fields
  const hasCreditCardField = Array.from(textFields).some(field => {
    const name = (field.name || field.id || '').toLowerCase();
    return name.includes('card') || name.includes('credit') || name.includes('cvv');
  });

  // Look for SSN fields
  const hasSSNField = Array.from(textFields).some(field => {
    const name = (field.name || field.id || '').toLowerCase();
    return name.includes('ssn') || name.includes('social');
  });

  // Calculate suspicion score
  let suspicionScore = 0;

  if (suspiciousCount > 5) suspicionScore += 20;
  if (passwordFields.length > 0 && !window.location.href.startsWith('https://')) {
    suspicionScore += 30;
  }
  if (hasCreditCardField && !window.location.href.startsWith('https://')) {
    suspicionScore += 40;
  }
  if (hasSSNField) suspicionScore += 30;

  // If suspicion score is high, show warning
  if (suspicionScore >= 50) {
    showInPageWarning(suspicionScore);
  }
}

/**
 * Show in-page warning
 */
function showInPageWarning(score) {
  // Don't show multiple warnings
  if (document.getElementById('phishing-shield-page-warning')) {
    return;
  }

  const warning = document.createElement('div');
  warning.id = 'phishing-shield-page-warning';
  warning.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    z-index: 999999;
    max-width: 350px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    animation: slideInRight 0.5s ease-out;
  `;

  warning.innerHTML = `
    <div style="display: flex; align-items: start; gap: 12px;">
      <span style="font-size: 32px; flex-shrink: 0;">⚠️</span>
      <div>
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">
          Phishing Warning
        </div>
        <div style="font-size: 13px; line-height: 1.5; opacity: 0.95; margin-bottom: 12px;">
          This page shows signs of phishing (Suspicion Score: ${score}/100).
          Be extremely careful before entering any personal information.
        </div>
        <div style="display: flex; gap: 8px;">
          <button id="phishing-shield-close-warning" style="
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
          ">
            Dismiss
          </button>
          <button id="phishing-shield-leave-page" style="
            background: white;
            border: none;
            color: #DC2626;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
          ">
            Leave Page
          </button>
        </div>
      </div>
    </div>
  `;

  // Add animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideInRight {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(warning);

  // Event listeners
  document.getElementById('phishing-shield-close-warning').addEventListener('click', () => {
    warning.remove();
  });

  document.getElementById('phishing-shield-leave-page').addEventListener('click', () => {
    window.history.back();
  });

  // Auto-dismiss after 30 seconds
  setTimeout(() => {
    if (warning.parentNode) {
      warning.remove();
    }
  }, 30000);
}

/**
 * Show notification
 */
function showWarningNotification(message) {
  // Create notification element
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #F59E0B;
    color: white;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    z-index: 999999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    animation: slideInDown 0.3s ease-out;
  `;

  notification.textContent = message;

  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideInDown {
      from { transform: translateY(-100%); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(notification);

  // Remove after 5 seconds
  setTimeout(() => {
    notification.remove();
  }, 5000);
}

/**
 * Detect SSL/TLS issues
 */
function checkSSL() {
  if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost') {
    // Check if page has password or sensitive fields
    const passwordFields = document.querySelectorAll('input[type="password"]');
    const emailFields = document.querySelectorAll('input[type="email"]');

    if (passwordFields.length > 0 || emailFields.length > 0) {
      console.warn('Phishing Shield: Insecure connection detected with sensitive fields');

      // Could show warning here
      showInPageWarning(70);
    }
  }
}

/**
 * Listen for messages from background script
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'showWarning') {
    showInPageWarning(message.score || 70);
    sendResponse({ success: true });
  }

  if (message.action === 'blockPage') {
    // Redirect to warning page
    window.location.href = chrome.runtime.getURL('warning.html') +
      `?url=${encodeURIComponent(window.location.href)}&score=${message.score || 100}`;
  }

  return true;
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Check SSL after page load
window.addEventListener('load', checkSSL);

console.log('Phishing Shield: Content script initialized');
