import { state, currentResults } from '../app/state.js';

const panel = document.getElementById('local-validation');
const text = document.getElementById('local-validation-text');
const button = document.getElementById('local-export');
const labels = { ours: '预测风场', cfd: 'CFD 参考', agnostic: '零风', profile: '剖面风' };

function summary() {
  const ready = currentResults() && !state.busy;
  button.disabled = !ready;
  if (!ready) {
    panel.dataset.status = 'pending';
    text.textContent = state.busy || state.cityLoading ? '正在计算；旧结果不可导出' : '等待当前任务的实时规划结果';
    return;
  }
  const rows = Object.entries(state.results.plans).map(([id, p]) => {
    if (!p.validation) return `${labels[id] ?? id}：${id === 'cfd' && state.results.scoredIn !== 'cfd' ? '此工况无 CFD 数据' : '未找到路线'}`;
    const c = p.validation.minClearanceM;
    const clearance = Number.isFinite(c) ? (c >= 10 ? '≥10.00' : c.toFixed(2)) : '未知';
    return `${labels[id] ?? id}：${p.validation.pass ? '通过' : '拒绝'} · 采样净空 ${clearance} m`;
  });
  const ours = state.results.plans.ours;
  panel.dataset.status = ours?.points && ours.validation?.pass ? 'passed' : 'failed';
  text.textContent = rows.join('\n') + '\n0.25 m 采样 · 建筑实体表面距离 · 非飞行安全认证';
}

export function report() {
  if (!currentResults() || state.busy) return null;
  const r = state.results;
  return {
    application: 'GeoWind2Plan Web Lab / Block A noncommercial adaptation',
    source: 'https://geowind2plan.com/', license: 'CC-BY-NC-4.0',
    generatedAt: new Date().toISOString(), computation: 'browser WebGPU + CPU Web Workers',
    city: state.cityEntry.id, mission: r.mission,
    wind: { speedMps: state.speed, directionRad: state.theta, modelSpeedScaling: 'U / 4 m/s' },
    scoring: r.scoredIn === 'cfd' ? 'offline quantized CFD at 4 m/s, 0 degrees' : 'no CFD truth; predicted scoring is not independent evaluation',
    precision: state.precision, gpu: state.gpuInfo, timing: r.timing,
    warning: 'Dense sampling only, not continuous collision proof or flight-dynamics certification.',
    plans: r.plans,
  };
}

window.addEventListener('geowind:results', summary);
button.addEventListener('click', () => {
  const value = report();
  if (!value) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'geowind-block-a-route-report.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
// A read-only diagnostics entry: never replaces compute with fixtures or precomputed trajectories.
Object.defineProperty(window, 'geoWindLab', { value: Object.freeze({
  report,
  status: () => ({ busy: state.busy, cityLoading: state.cityLoading, current: Boolean(currentResults()),
    precision: state.precision, gpu: state.gpuInfo, hasPrediction: Boolean(state.prediction),
    predictionMilliseconds: state.prediction?.milliseconds, hasResults: Boolean(state.results),
    failure: state.failureShown, mode: state.mode }),
}) });
setInterval(summary, 400);
summary();
