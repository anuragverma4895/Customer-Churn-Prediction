/**
 * Main Application Logic for Customer Churn Dashboard.
 * Fetches data, populates UI, initializes charts, handles predictions.
 */

const App = (() => {
  const API_BASE = '/api';
  let cachedData = {};

  /* ─── Fetch Helpers ─── */
  async function fetchJSON(endpoint) {
    try {
      const res = await fetch(`${API_BASE}/${endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error(`Failed to fetch ${endpoint}:`, err);
      return null;
    }
  }

  /* ─── Animated Counter ─── */
  function animateCounter(elementId, targetValue, suffix = '', duration = 1500, decimals = 0) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (targetValue - start) * eased;

      el.textContent = decimals > 0
        ? current.toFixed(decimals) + suffix
        : Math.round(current).toLocaleString() + suffix;

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }

  /* ─── Populate KPI Cards ─── */
  function populateKPICards(stats, metrics) {
    if (!stats || !metrics) return;

    // Total Customers
    animateCounter('kpi-total-customers', stats.total_customers, '', 1500);

    // Churn Rate
    animateCounter('kpi-churn-rate', stats.churn_rate, '%', 1200, 1);

    // Best Model Accuracy (use best model)
    const bestKey = metrics.best_model === 'Logistic Regression' ? 'logistic_regression' : 'random_forest';
    const bestMetrics = metrics[bestKey];
    animateCounter('kpi-accuracy', bestMetrics.accuracy * 100, '%', 1400, 1);

    // F1 Score
    animateCounter('kpi-f1', bestMetrics.f1 * 100, '%', 1300, 1);
  }

  /* ─── Model Comparison Table ─── */
  function populateModelComparison(metrics) {
    if (!metrics) return;

    const lr = metrics.logistic_regression;
    const rf = metrics.random_forest;
    const bestModel = metrics.best_model;

    const metricNames = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
    const tbody = document.getElementById('comparison-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';

    metricNames.forEach(metric => {
      const lrVal = lr[metric];
      const rfVal = rf[metric];
      const lrBetter = lrVal > rfVal;
      const rfBetter = rfVal > lrVal;

      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="metric-name">${metric.charAt(0).toUpperCase() + metric.slice(1)}</td>
        <td class="${lrBetter ? 'metric-winner' : ''}">${(lrVal * 100).toFixed(2)}%${lrBetter ? ' <span class="winner-badge">' + Icons.award(14) + '</span>' : ''}</td>
        <td class="${rfBetter ? 'metric-winner' : ''}">${(rfVal * 100).toFixed(2)}%${rfBetter ? ' <span class="winner-badge">' + Icons.award(14) + '</span>' : ''}</td>
      `;
      tbody.appendChild(row);
    });

    // Update best model badge
    const bestBadge = document.getElementById('best-model-badge');
    if (bestBadge) {
      bestBadge.innerHTML = `${Icons.award(16)} <span>Best: ${bestModel}</span>`;
    }
  }

  /* ─── Chart Model Selector ─── */
  function setupChartSelectors() {
    const cmSelector = document.getElementById('cm-model-select');
    const fiSelector = document.getElementById('fi-model-select');

    if (cmSelector) {
      cmSelector.addEventListener('change', (e) => {
        if (cachedData.confusionMatrix) {
          ChartManager.renderConfusionMatrix('cm-chart', cachedData.confusionMatrix, e.target.value);
        }
      });
    }

    if (fiSelector) {
      fiSelector.addEventListener('change', (e) => {
        if (cachedData.featureImportance) {
          ChartManager.renderFeatureImportance('fi-chart', cachedData.featureImportance, e.target.value);
        }
      });
    }
  }

  /* ─── Prediction Form ─── */
  function setupPredictionForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const submitBtn = form.querySelector('button[type="submit"]');
      const resultDiv = document.getElementById('prediction-result');

      // Collect form data
      const formData = new FormData(form);
      const customerData = {};
      formData.forEach((value, key) => {
        // Convert numeric fields
        if (['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'].includes(key)) {
          customerData[key] = parseFloat(value);
        } else {
          customerData[key] = value;
        }
      });

      // Show loading state
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span class="spinner"></span> Analyzing...`;
      resultDiv.classList.remove('visible');

      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(customerData),
        });

        const result = await res.json();

        if (result.error) {
          showPredictionError(resultDiv, result.error);
        } else {
          showPredictionResult(resultDiv, result);
        }
      } catch (err) {
        showPredictionError(resultDiv, 'Failed to connect to the prediction API.');
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `${Icons.send(16)} <span>Predict Churn</span>`;
      }
    });
  }

  function showPredictionResult(container, result) {
    const prob = (result.churn_probability * 100).toFixed(1);
    const risk = result.risk_level;

    let riskClass, riskIcon;
    if (risk === 'Low') {
      riskClass = 'risk-low';
      riskIcon = Icons.checkCircle(24);
    } else if (risk === 'Medium') {
      riskClass = 'risk-medium';
      riskIcon = Icons.alertTriangle(24);
    } else {
      riskClass = 'risk-high';
      riskIcon = Icons.xCircle(24);
    }

    container.innerHTML = `
      <div class="prediction-card ${riskClass}">
        <div class="risk-header">
          <span class="risk-icon">${riskIcon}</span>
          <span class="risk-label">${risk} Risk</span>
        </div>
        <div class="risk-gauge">
          <div class="gauge-track">
            <div class="gauge-fill" style="width: ${prob}%"></div>
          </div>
          <div class="gauge-labels">
            <span>0%</span>
            <span class="gauge-value">${prob}%</span>
            <span>100%</span>
          </div>
        </div>
        <div class="prediction-details">
          <div class="detail-item">
            <span class="detail-label">Churn Probability</span>
            <span class="detail-value">${prob}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Prediction</span>
            <span class="detail-value">${result.churn_label === 'Yes' ? 'Will Churn' : 'Will Stay'}</span>
          </div>
        </div>
      </div>
    `;
    container.classList.add('visible');

    // Animate gauge fill
    requestAnimationFrame(() => {
      const fill = container.querySelector('.gauge-fill');
      if (fill) {
        fill.style.transition = 'width 1s ease-out';
        fill.style.width = `${prob}%`;
      }
    });
  }

  function showPredictionError(container, message) {
    container.innerHTML = `
      <div class="prediction-card risk-error">
        <div class="risk-header">
          <span class="risk-icon">${Icons.alertTriangle(24)}</span>
          <span class="risk-label">Error</span>
        </div>
        <p class="error-message">${message}</p>
      </div>
    `;
    container.classList.add('visible');
  }

  /* ─── Intersection Observer for Animations ─── */
  function setupScrollAnimations() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    );

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
      observer.observe(el);
    });
  }

  /* ─── Initialize All ─── */
  async function init() {
    // Set up UI elements
    setupChartSelectors();
    setupPredictionForm();
    setupScrollAnimations();

    // Inject icons into KPI cards
    injectIcons();

    // Fetch all data in parallel
    const [metrics, featureImportance, confusionMatrix, rocData, datasetStats] = await Promise.all([
      fetchJSON('metrics'),
      fetchJSON('feature-importance'),
      fetchJSON('confusion-matrix'),
      fetchJSON('roc-curve'),
      fetchJSON('dataset-stats'),
    ]);

    // Cache data for chart model switching
    cachedData = { metrics, featureImportance, confusionMatrix, rocData, datasetStats };

    // Populate UI
    populateKPICards(datasetStats, metrics);
    populateModelComparison(metrics);

    // Render charts
    if (confusionMatrix) {
      ChartManager.renderConfusionMatrix('cm-chart', confusionMatrix, 'random_forest');
    }
    if (rocData) {
      ChartManager.renderROCCurve('roc-chart', rocData);
    }
    if (featureImportance) {
      ChartManager.renderFeatureImportance('fi-chart', featureImportance, 'random_forest');
    }
    if (datasetStats) {
      ChartManager.renderChurnDistribution('dist-chart', datasetStats);
    }

    // Theme change handler — re-render charts
    ThemeManager.onChartUpdate(() => {
      ChartManager.destroyAll();
      if (confusionMatrix) {
        const cmModel = document.getElementById('cm-model-select')?.value || 'random_forest';
        ChartManager.renderConfusionMatrix('cm-chart', confusionMatrix, cmModel);
      }
      if (rocData) ChartManager.renderROCCurve('roc-chart', rocData);
      if (featureImportance) {
        const fiModel = document.getElementById('fi-model-select')?.value || 'random_forest';
        ChartManager.renderFeatureImportance('fi-chart', featureImportance, fiModel);
      }
      if (datasetStats) ChartManager.renderChurnDistribution('dist-chart', datasetStats);
    });

    // Remove loading state
    document.body.classList.add('loaded');
  }

  function injectIcons() {
    // KPI card icons
    const iconMap = {
      'kpi-icon-customers': Icons.users(22),
      'kpi-icon-churn': Icons.trendingDown(22),
      'kpi-icon-accuracy': Icons.target(22),
      'kpi-icon-f1': Icons.activity(22),
    };

    Object.entries(iconMap).forEach(([id, svg]) => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = svg;
    });
  }

  return { init };
})();

// Boot
document.addEventListener('DOMContentLoaded', () => {
  ThemeManager.init();
  App.init();
});
