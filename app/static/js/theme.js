/**
 * Theme Toggle - Dark / Light mode switcher.
 * Persists preference in localStorage.
 * Updates CSS custom properties and Chart.js colors.
 */

const ThemeManager = (() => {
  const STORAGE_KEY = 'churn-dashboard-theme';
  let currentTheme = 'dark';
  let chartUpdateCallback = null;

  function init() {
    const saved = localStorage.getItem(STORAGE_KEY);
    currentTheme = saved || 'dark';
    applyTheme(currentTheme);
    setupToggleButton();
  }

  function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);

    // Update toggle button icon
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      toggleBtn.innerHTML = theme === 'dark' ? Icons.sun(18) : Icons.moon(18);
      toggleBtn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    }

    // Notify charts to update colors
    if (chartUpdateCallback) {
      chartUpdateCallback(theme);
    }
  }

  function setupToggleButton() {
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
      });
    }
  }

  function onChartUpdate(callback) {
    chartUpdateCallback = callback;
  }

  function getTheme() {
    return currentTheme;
  }

  function getColors() {
    const root = getComputedStyle(document.documentElement);
    return {
      bgPrimary: root.getPropertyValue('--bg-primary').trim(),
      bgSurface: root.getPropertyValue('--bg-surface').trim(),
      bgSurfaceHover: root.getPropertyValue('--bg-surface-hover').trim(),
      borderColor: root.getPropertyValue('--border-color').trim(),
      textPrimary: root.getPropertyValue('--text-primary').trim(),
      textSecondary: root.getPropertyValue('--text-secondary').trim(),
      accentBlue: root.getPropertyValue('--accent-blue').trim(),
      accentGreen: root.getPropertyValue('--accent-green').trim(),
      accentAmber: root.getPropertyValue('--accent-amber').trim(),
      accentRed: root.getPropertyValue('--accent-red').trim(),
      accentPurple: root.getPropertyValue('--accent-purple').trim(),
    };
  }

  return { init, getTheme, getColors, onChartUpdate };
})();
