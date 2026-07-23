/**
 * Main application logic for the Customer Churn dashboard.
 * Loads exported metrics, real CSV examples, charts, and prediction results.
 */

const App = (() => {
  const API_BASE = '/api';
  const MODEL_LABELS = {
    logistic_regression: 'Logistic Regression',
    random_forest: 'Random Forest',
  };
  const NUMERIC_FIELDS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'];

  let cachedData = {};
  let selectedExample = null;

  async function fetchJSON(endpoint) {
    try {
      const res = await fetch(`${API_BASE}/${endpoint}`);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      return data;
    } catch (err) {
      console.error(`Failed to fetch ${endpoint}:`, err);
      return null;
    }
  }

  function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>'"]/g, (char) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;',
    }[char]));
  }

  function getBestModelKey(metrics) {
    return metrics?.best_model === 'Random Forest' ? 'random_forest' : 'logistic_regression';
  }

  function formatPercent(value, decimals = 1) {
    return `${(Number(value) * 100).toFixed(decimals)}%`;
  }

  function animateCounter(elementId, targetValue, suffix = '', duration = 1200, decimals = 0) {
    const el = document.getElementById(elementId);
    if (!el || Number.isNaN(Number(targetValue))) return;

    const startTime = performance.now();
    const target = Number(targetValue);

    function update(currentTime) {
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;
      el.textContent = decimals > 0
        ? `${current.toFixed(decimals)}${suffix}`
        : `${Math.round(current).toLocaleString()}${suffix}`;
      if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function populateKPICards(stats, metrics) {
    if (!stats || !metrics) return;

    const bestKey = getBestModelKey(metrics);
    const bestMetrics = metrics[bestKey];

    animateCounter('kpi-total-customers', stats.total_customers);
    animateCounter('kpi-churn-rate', stats.churn_rate, '%', 1100, 1);
    animateCounter('kpi-accuracy', bestMetrics.accuracy * 100, '%', 1200, 1);
    animateCounter('kpi-f1', bestMetrics.f1 * 100, '%', 1200, 1);

    setText('kpi-churn-count', `${stats.churn_yes.toLocaleString()} of ${stats.total_customers.toLocaleString()}`);
    setText('kpi-accuracy-note', MODEL_LABELS[bestKey]);
    setText('source-pill', `Real dataset: ${stats.total_customers.toLocaleString()} IBM Telco rows`);
    setText('best-model-pill', `Best model: ${MODEL_LABELS[bestKey]}`);
  }

  function populateDatasetInsights(stats) {
    const container = document.getElementById('dataset-insights');
    if (!container || !stats) return;

    const largestContract = topCategory(stats.contract_distribution);
    const largestPayment = topCategory(stats.payment_distribution);
    container.innerHTML = `
      <div class="insight-row"><span>Contract mix</span><strong>${escapeHTML(largestContract.label)} (${largestContract.value.toLocaleString()})</strong></div>
      <div class="insight-row"><span>Payment mix</span><strong>${escapeHTML(largestPayment.label)} (${largestPayment.value.toLocaleString()})</strong></div>
      <div class="insight-row"><span>Average monthly charge</span><strong>$${stats.monthly_charges_stats.mean.toFixed(2)}</strong></div>
      <div class="insight-row"><span>Median tenure</span><strong>${stats.tenure_stats.median} months</strong></div>
    `;
  }

  function topCategory(distribution) {
    const [label, value] = Object.entries(distribution || {}).sort((a, b) => b[1] - a[1])[0] || ['N/A', 0];
    return { label, value };
  }

  function populateModelComparison(metrics) {
    if (!metrics) return;

    const lr = metrics.logistic_regression;
    const rf = metrics.random_forest;
    const metricNames = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
    const tbody = document.getElementById('comparison-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    metricNames.forEach((metric) => {
      const lrVal = lr[metric];
      const rfVal = rf[metric];
      const lrBetter = lrVal > rfVal;
      const rfBetter = rfVal > lrVal;
      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="metric-name">${metric.toUpperCase()}</td>
        <td class="${lrBetter ? 'metric-winner' : ''}">${formatPercent(lrVal, 2)}${lrBetter ? ` <span class="winner-badge">${Icons.award(14)}</span>` : ''}</td>
        <td class="${rfBetter ? 'metric-winner' : ''}">${formatPercent(rfVal, 2)}${rfBetter ? ` <span class="winner-badge">${Icons.award(14)}</span>` : ''}</td>
      `;
      tbody.appendChild(row);
    });

    const bestBadge = document.getElementById('best-model-badge');
    if (bestBadge) bestBadge.innerHTML = `${Icons.award(16)} <span>Best: ${escapeHTML(metrics.best_model)}</span>`;
  }

  function setupChartSelectors() {
    const cmSelector = document.getElementById('cm-model-select');
    const fiSelector = document.getElementById('fi-model-select');

    cmSelector?.addEventListener('change', (e) => {
      if (cachedData.confusionMatrix) ChartManager.renderConfusionMatrix('cm-chart', cachedData.confusionMatrix, e.target.value);
    });

    fiSelector?.addEventListener('change', (e) => {
      if (cachedData.featureImportance) ChartManager.renderFeatureImportance('fi-chart', cachedData.featureImportance, e.target.value);
    });
  }

  function populateRealExamples(payload) {
    const select = document.getElementById('real-customer-select');
    const meta = document.getElementById('record-meta');
    if (!select) return;

    const examples = payload?.examples || [];
    if (!examples.length) {
      select.innerHTML = '<option value="">No real records available</option>';
      if (meta) meta.textContent = 'Run the training pipeline to restore the local CSV.';
      return;
    }

    select.innerHTML = '';
    examples.forEach((example, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${example.label} - ${example.customer_id} - Actual churn: ${example.actual_churn}`;
      select.appendChild(option);
    });

    select.addEventListener('change', () => applyExample(examples[Number(select.value)]));
    applyExample(examples[0]);
  }

  function applyExample(example) {
    if (!example) return;
    selectedExample = example;
    setFormValues(example.customer_data);

    const meta = document.getElementById('record-meta');
    if (meta) {
      meta.innerHTML = `Customer ID <strong>${escapeHTML(example.customer_id)}</strong> | Actual churn <strong>${escapeHTML(example.actual_churn)}</strong>`;
    }
  }

  function setFormValues(values) {
    Object.entries(values || {}).forEach(([key, value]) => {
      const field = document.querySelector(`[name="${CSS.escape(key)}"]`);
      if (!field) return;
      field.value = String(value);
    });
  }

  function getCustomerData(form) {
    const formData = new FormData(form);
    const customerData = {};
    formData.forEach((value, key) => {
      customerData[key] = NUMERIC_FIELDS.includes(key) ? Number(value) : value;
    });
    return customerData;
  }

  function setupPredictionForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('input', () => {
      selectedExample = null;
      const meta = document.getElementById('record-meta');
      if (meta) meta.textContent = 'Manual profile based on Telco dataset fields';
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = form.querySelector('button[type="submit"]');
      const resultDiv = document.getElementById('prediction-result');
      const customerData = getCustomerData(form);

      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span class="spinner"></span> Analyzing customer...`;
      resultDiv.classList.remove('visible');

      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(customerData),
        });
        const result = await res.json();
        if (result.error) showPredictionError(resultDiv, result.error);
        else showPredictionResult(resultDiv, result, customerData);
      } catch (err) {
        showPredictionError(resultDiv, 'Prediction API is not reachable. Start the Flask server and try again.');
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `${Icons.send(16)} <span>Predict Churn Risk</span>`;
      }
    });
  }

  function showPredictionResult(container, result, customerData) {
    const probability = Number(result.churn_probability);
    const probPct = (probability * 100).toFixed(1);
    const risk = result.risk_level;
    const riskClass = risk === 'Low' ? 'risk-low' : risk === 'Medium' ? 'risk-medium' : 'risk-high';
    const riskIcon = risk === 'Low' ? Icons.checkCircle(24) : risk === 'Medium' ? Icons.alertTriangle(24) : Icons.xCircle(24);
    const actions = buildRetentionActions(risk, customerData);
    const actualChurn = selectedExample ? selectedExample.actual_churn : 'Manual profile';

    container.className = 'visible';
    container.innerHTML = `
      <div class="prediction-card ${riskClass}">
        <div class="risk-header">
          <span class="risk-icon">${riskIcon}</span>
          <div>
            <span class="risk-label">${escapeHTML(risk)} Risk</span>
            <small>${MODEL_LABELS[result.model] || result.model}</small>
          </div>
        </div>
        <div class="risk-gauge">
          <div class="gauge-track"><div class="gauge-fill" style="width: 0%"></div></div>
          <div class="gauge-labels"><span>0%</span><span class="gauge-value">${probPct}% churn probability</span><span>100%</span></div>
        </div>
        <div class="prediction-details">
          <div class="detail-item"><span class="detail-label">Prediction</span><span class="detail-value">${result.churn_label === 'Yes' ? 'Likely to churn' : 'Likely to stay'}</span></div>
          <div class="detail-item"><span class="detail-label">Actual record churn</span><span class="detail-value">${escapeHTML(actualChurn)}</span></div>
          <div class="detail-item"><span class="detail-label">Contract</span><span class="detail-value">${escapeHTML(customerData.Contract)}</span></div>
          <div class="detail-item"><span class="detail-label">Monthly charge</span><span class="detail-value">$${Number(customerData.MonthlyCharges).toFixed(2)}</span></div>
        </div>
        <div class="recommendation-box">
          <span>Retention focus</span>
          <ul>${actions.map((item) => `<li>${escapeHTML(item)}</li>`).join('')}</ul>
        </div>
      </div>
    `;

    requestAnimationFrame(() => {
      const fill = container.querySelector('.gauge-fill');
      if (fill) fill.style.width = `${probPct}%`;
    });
  }

  function buildRetentionActions(risk, data) {
    const actions = [];

    if (risk === 'High') actions.push('Prioritize outreach before the next billing cycle.');
    if (risk === 'Medium') actions.push('Review service fit and offer a targeted save plan.');
    if (risk === 'Low') actions.push('Keep engagement steady and monitor plan changes.');
    if (data.Contract === 'Month-to-month') actions.push('Offer an annual contract incentive.');
    if (data.TechSupport === 'No' && data.InternetService !== 'No') actions.push('Bundle tech support to reduce service friction.');
    if (data.PaymentMethod === 'Electronic check') actions.push('Encourage automatic payment to improve retention.');
    if (Number(data.tenure) <= 6) actions.push('Add onboarding follow-up for early-tenure risk.');

    return [...new Set(actions)].slice(0, 4);
  }

  function showPredictionError(container, message) {
    container.className = 'visible';
    container.innerHTML = `
      <div class="prediction-card risk-error">
        <div class="risk-header"><span class="risk-icon">${Icons.alertTriangle(24)}</span><span class="risk-label">Error</span></div>
        <p class="error-message">${escapeHTML(message)}</p>
      </div>
    `;
  }

  function setupScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-in');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.animate-on-scroll').forEach((el) => observer.observe(el));
  }

  function injectIcons() {
    const iconMap = {
      'kpi-icon-customers': Icons.users(22),
      'kpi-icon-churn': Icons.trendingDown(22),
      'kpi-icon-accuracy': Icons.target(22),
      'kpi-icon-f1': Icons.activity(22),
      'empty-result-icon': Icons.gauge(34),
      'github-link': Icons.github(18),
      'submit-icon': Icons.send(16),
    };

    Object.entries(iconMap).forEach(([id, svg]) => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = svg;
    });

    const brandIcon = document.querySelector('.brand-icon');
    if (brandIcon) brandIcon.innerHTML = Icons.activity(18);
  }

  function renderCharts(defaultKey) {
    const cmKey = document.getElementById('cm-model-select')?.value || defaultKey;
    const fiKey = document.getElementById('fi-model-select')?.value || defaultKey;
    if (cachedData.confusionMatrix) ChartManager.renderConfusionMatrix('cm-chart', cachedData.confusionMatrix, cmKey);
    if (cachedData.rocData) ChartManager.renderROCCurve('roc-chart', cachedData.rocData);
    if (cachedData.featureImportance) ChartManager.renderFeatureImportance('fi-chart', cachedData.featureImportance, fiKey);
    if (cachedData.datasetStats) ChartManager.renderChurnDistribution('dist-chart', cachedData.datasetStats);
  }

  async function init() {
    setupChartSelectors();
    setupPredictionForm();
    setupScrollAnimations();
    injectIcons();

    const [metrics, featureImportance, confusionMatrix, rocData, datasetStats, examples] = await Promise.all([
      fetchJSON('metrics'),
      fetchJSON('feature-importance'),
      fetchJSON('confusion-matrix'),
      fetchJSON('roc-curve'),
      fetchJSON('dataset-stats'),
      fetchJSON('customer-examples'),
    ]);

    cachedData = { metrics, featureImportance, confusionMatrix, rocData, datasetStats, examples };
    const bestKey = getBestModelKey(metrics);

    document.getElementById('model').value = bestKey;
    document.getElementById('cm-model-select').value = bestKey;
    document.getElementById('fi-model-select').value = bestKey;

    populateKPICards(datasetStats, metrics);
    populateDatasetInsights(datasetStats);
    populateModelComparison(metrics);
    populateRealExamples(examples);
    renderCharts(bestKey);

    ThemeManager.onChartUpdate(() => {
      ChartManager.destroyAll();
      renderCharts(bestKey);
    });

    document.body.classList.add('loaded');
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', () => {
  ThemeManager.init();
  App.init();
});
