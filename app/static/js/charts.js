/**
 * Chart.js Visualizations for Customer Churn Dashboard.
 * Confusion Matrix, ROC Curve, Feature Importance, Churn Distribution.
 */

const ChartManager = (() => {
  let charts = {};

  function getChartColors() {
    return ThemeManager.getColors();
  }

  function destroyAll() {
    Object.values(charts).forEach(chart => {
      if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
      }
    });
    charts = {};
  }

  /* ─── Confusion Matrix ─── */
  function renderConfusionMatrix(canvasId, data, modelKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = getChartColors();
    const matrix = data[modelKey].matrix;
    const labels = data[modelKey].labels;

    // Flatten matrix for chart
    const tn = matrix[0][0], fp = matrix[0][1];
    const fn = matrix[1][0], tp = matrix[1][1];
    const total = tn + fp + fn + tp;

    if (charts[canvasId]) charts[canvasId].destroy();

    charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
        datasets: [{
          data: [tn, fp, fn, tp],
          backgroundColor: [
            colors.accentGreen + 'cc',
            colors.accentAmber + 'cc',
            colors.accentRed + 'cc',
            colors.accentBlue + 'cc',
          ],
          borderColor: [
            colors.accentGreen,
            colors.accentAmber,
            colors.accentRed,
            colors.accentBlue,
          ],
          borderWidth: 1,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.bgSurface,
            titleColor: colors.textPrimary,
            bodyColor: colors.textSecondary,
            borderColor: colors.borderColor,
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            callbacks: {
              label: (context) => {
                const val = context.raw;
                const pct = ((val / total) * 100).toFixed(1);
                return `Count: ${val} (${pct}%)`;
              }
            }
          }
        },
        scales: {
          x: {
            ticks: { color: colors.textSecondary, font: { size: 11 } },
            grid: { display: false },
          },
          y: {
            ticks: { color: colors.textSecondary },
            grid: { color: colors.borderColor + '40' },
          }
        },
        animation: {
          duration: 1200,
          easing: 'easeOutQuart',
        }
      }
    });
  }

  /* ─── ROC Curve ─── */
  function renderROCCurve(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = getChartColors();
    const lrData = data.logistic_regression;
    const rfData = data.random_forest;

    // Downsample ROC points for performance
    const downsample = (fpr, tpr, maxPoints = 100) => {
      if (fpr.length <= maxPoints) {
        return fpr.map((f, i) => ({ x: f, y: tpr[i] }));
      }
      const step = Math.ceil(fpr.length / maxPoints);
      const points = [];
      for (let i = 0; i < fpr.length; i += step) {
        points.push({ x: fpr[i], y: tpr[i] });
      }
      // Always include last point
      points.push({ x: fpr[fpr.length - 1], y: tpr[tpr.length - 1] });
      return points;
    };

    if (charts[canvasId]) charts[canvasId].destroy();

    charts[canvasId] = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: `Logistic Regression (AUC: ${lrData.auc.toFixed(3)})`,
            data: downsample(lrData.fpr, lrData.tpr),
            showLine: true,
            borderColor: colors.accentBlue,
            backgroundColor: colors.accentBlue + '18',
            fill: true,
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.3,
          },
          {
            label: `Random Forest (AUC: ${rfData.auc.toFixed(3)})`,
            data: downsample(rfData.fpr, rfData.tpr),
            showLine: true,
            borderColor: colors.accentGreen,
            backgroundColor: colors.accentGreen + '18',
            fill: true,
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.3,
          },
          {
            label: 'Random Baseline',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            showLine: true,
            borderColor: colors.textSecondary + '60',
            borderWidth: 1.5,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: colors.textSecondary,
              padding: 16,
              usePointStyle: true,
              pointStyleWidth: 12,
              font: { size: 11 },
            }
          },
          tooltip: {
            backgroundColor: colors.bgSurface,
            titleColor: colors.textPrimary,
            bodyColor: colors.textSecondary,
            borderColor: colors.borderColor,
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'False Positive Rate', color: colors.textSecondary, font: { size: 12 } },
            min: 0, max: 1,
            ticks: { color: colors.textSecondary },
            grid: { color: colors.borderColor + '30' },
          },
          y: {
            title: { display: true, text: 'True Positive Rate', color: colors.textSecondary, font: { size: 12 } },
            min: 0, max: 1,
            ticks: { color: colors.textSecondary },
            grid: { color: colors.borderColor + '30' },
          }
        },
        animation: {
          duration: 1500,
          easing: 'easeOutQuart',
        }
      }
    });
  }

  /* ─── Feature Importance ─── */
  function renderFeatureImportance(canvasId, data, modelKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = getChartColors();
    const features = data[modelKey].slice(0, 10).reverse();

    // Generate gradient colors
    const barColors = features.map((_, i) => {
      const ratio = i / (features.length - 1);
      // Gradient from accent-red (low) → accent-blue (high)
      return `hsl(${200 + ratio * 20}, 70%, ${50 + ratio * 10}%)`;
    });

    if (charts[canvasId]) charts[canvasId].destroy();

    charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: features.map(f => f.feature.replace(/_/g, ' ')),
        datasets: [{
          data: features.map(f => f.importance),
          backgroundColor: barColors.map(c => c.replace(')', ', 0.8)').replace('hsl', 'hsla')),
          borderColor: barColors,
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.bgSurface,
            titleColor: colors.textPrimary,
            bodyColor: colors.textSecondary,
            borderColor: colors.borderColor,
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            callbacks: {
              label: (context) => `Importance: ${(context.raw * 100).toFixed(1)}%`
            }
          }
        },
        scales: {
          x: {
            min: 0, max: 1,
            ticks: { color: colors.textSecondary, callback: (v) => `${(v * 100).toFixed(0)}%` },
            grid: { color: colors.borderColor + '30' },
          },
          y: {
            ticks: { color: colors.textSecondary, font: { size: 11 } },
            grid: { display: false },
          }
        },
        animation: {
          duration: 1400,
          easing: 'easeOutQuart',
        }
      }
    });
  }

  /* ─── Churn Distribution Donut ─── */
  function renderChurnDistribution(canvasId, stats) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = getChartColors();

    if (charts[canvasId]) charts[canvasId].destroy();

    charts[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Retained', 'Churned'],
        datasets: [{
          data: [stats.churn_no, stats.churn_yes],
          backgroundColor: [
            colors.accentGreen + 'dd',
            colors.accentRed + 'dd',
          ],
          borderColor: [
            colors.accentGreen,
            colors.accentRed,
          ],
          borderWidth: 2,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: colors.textSecondary,
              padding: 20,
              usePointStyle: true,
              pointStyleWidth: 12,
              font: { size: 12 },
            }
          },
          tooltip: {
            backgroundColor: colors.bgSurface,
            titleColor: colors.textPrimary,
            bodyColor: colors.textSecondary,
            borderColor: colors.borderColor,
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            callbacks: {
              label: (context) => {
                const total = stats.churn_no + stats.churn_yes;
                const pct = ((context.raw / total) * 100).toFixed(1);
                return `${context.label}: ${context.raw.toLocaleString()} (${pct}%)`;
              }
            }
          }
        },
        animation: {
          animateRotate: true,
          duration: 1200,
          easing: 'easeOutQuart',
        }
      }
    });
  }

  /* ─── Theme Update ─── */
  function updateAllCharts() {
    // Re-render all charts with new theme colors
    // This is handled by the data refetch in app.js
  }

  return {
    renderConfusionMatrix,
    renderROCCurve,
    renderFeatureImportance,
    renderChurnDistribution,
    destroyAll,
    updateAllCharts,
  };
})();
