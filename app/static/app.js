/* VancoDose front end: vanilla JS, no build step. */
const C = { hebe: '#B36CFF', purple: '#B36CFF', violet: '#844BFF', violet700: '#6A34E6', ultra: '#5E6BFF', river: '#43C0FF',
  ink: '#1B1838', muted: '#75718C', line: '#E8E5F2', below: '#C77A0E', within: '#0F8A61', above: '#D9601F', toxic: '#CF3343' };
const CAT = { below: 'Below target', within: 'Within target', above: 'Above target', toxic: 'Toxicity concern' };
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (x, d = 0) => (x == null || Number.isNaN(+x)) ? '—' : Number(x).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sign = x => (x > 0 ? '+' : '') + fmt(x);
const tag = (cat, text) => `<span class="tag ${cat}">${esc(text ?? CAT[cat])}</span>`;
const charts = {};
let examples = [];
let analyticsLoaded = false, modelLoaded = false;

async function api(url, opts = {}) {
  const res = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed. Check the values and try again.');
  return data;
}

/* ---------------------------------------------------------------- chart helpers */
const bandPlugin = { id: 'band', beforeDatasetsDraw(chart, _a, o) {
  if (!o?.bands) return; const { ctx, chartArea: a, scales } = chart;
  for (const b of o.bands) { const s = scales[b.axis || 'y']; if (!s) continue;
    const y1 = s.getPixelForValue(b.from), y2 = s.getPixelForValue(b.to);
    ctx.save(); ctx.fillStyle = b.color; ctx.fillRect(a.left, Math.max(Math.min(y1, y2), a.top), a.right - a.left, Math.min(Math.abs(y2 - y1), a.bottom - a.top));
    if (b.label) { ctx.fillStyle = b.text || C.muted; ctx.font = '600 11px Manrope, sans-serif'; ctx.fillText(b.label, a.left + 6, Math.max(Math.min(y1, y2), a.top) + 13); }
    ctx.restore(); } } };
const vlinePlugin = { id: 'vlines', afterDatasetsDraw(chart, _a, o) {
  if (!o?.lines) return; const { ctx, chartArea: a, scales: { x } } = chart;
  for (const l of o.lines) { const px = x.getPixelForValue(l.x); if (px < a.left || px > a.right) continue;
    ctx.save(); ctx.strokeStyle = l.color || C.line; ctx.setLineDash(l.dash || [4, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = l.text || C.muted; ctx.font = '700 11px Manrope, sans-serif'; ctx.fillText(l.label, px + 4, a.top + 12); ctx.restore(); } } };
const baseOpts = () => ({ animation: false, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, padding: 14 } } } });
const mkChart = (id, cfg) => { charts[id]?.destroy(); const el = document.getElementById(id); if (el && window.Chart) charts[id] = new Chart(el, cfg); };

/* ---------------------------------------------------------------- navigation */
document.addEventListener('DOMContentLoaded', () => {
  if (window.Chart) { Chart.defaults.font.family = 'Manrope, sans-serif'; Chart.defaults.color = C.muted; Chart.defaults.maintainAspectRatio = false; Chart.register(bandPlugin, vlinePlugin); }
  $$('.tab').forEach(b => b.addEventListener('click', () => go(b.dataset.view)));
  $('.brand').addEventListener('click', e => { e.preventDefault(); go('patients'); });
  $('#help-btn').addEventListener('click', toggleGlossary);
  $('#gl-close').addEventListener('click', toggleGlossary);
  document.addEventListener('keydown', e => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName);
    if (e.key === 'Escape') { if (!$('#glossary').hidden) toggleGlossary(); closeList(); }
    if (typing) return;
    if (e.key === '/') { e.preventDefault(); go('patients'); $('#pt-search').focus(); }
    if (e.key === '?') { e.preventDefault(); toggleGlossary(); }
  });
  window.addEventListener('hashchange', route);
  initForm();
  initPatients();
  route();
});

function go(view, id) { location.hash = id ? `${view}/${id}` : view; }
function route() {
  const [view = 'patients', id] = location.hash.replace('#', '').split('/');
  const v = ['patients', 'dosing', 'analytics', 'model'].includes(view) ? view : 'patients';
  $$('.tab').forEach(b => { b.classList.toggle('active', b.dataset.view === v); b.setAttribute('aria-current', b.dataset.view === v ? 'page' : 'false'); });
  $$('.view').forEach(s => s.classList.toggle('active', s.id === 'view-' + v));
  if (v === 'analytics' && !analyticsLoaded) loadAnalytics();
  if (v === 'model' && !modelLoaded) loadModel();
  if (v === 'patients' && id && id !== state.patient?.patient.id) loadPatient(decodeURIComponent(id));
  if (v === 'patients' && !id) { state.patient = null; $('#pt-view').hidden = true; $('#pt-empty').hidden = false; $('#pt-search').value = ''; }
}
function toggleGlossary() { const g = $('#glossary'); g.hidden = !g.hidden; if (!g.hidden) $('#gl-close').focus(); }
let toastTimer;
function toast(msg, action) {
  const t = $('#toast'); t.innerHTML = `<span>${esc(msg)}</span>${action ? `<button type="button">${esc(action.label)}</button>` : ''}`; t.hidden = false;
  if (action) t.querySelector('button').onclick = () => { t.hidden = true; action.run(); };
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 7000);
}

/* ================================================================ PATIENTS */
const state = { directory: [], patient: null, selected: 0, active: -1 };
const recentKey = 'vancodose.recent';
const getRecent = () => { try { return JSON.parse(localStorage.getItem(recentKey) || '[]'); } catch { return []; } };
const pushRecent = id => { try { localStorage.setItem(recentKey, JSON.stringify([id, ...getRecent().filter(x => x !== id)].slice(0, 5))); } catch { /* storage unavailable */ } };

async function initPatients() {
  const input = $('#pt-search');
  input.addEventListener('input', () => { state.active = 0; openList(); });
  input.addEventListener('focus', openList);
  input.addEventListener('keydown', e => {
    const items = $$('#pt-list li[data-id]');
    if (e.key === 'ArrowDown') { e.preventDefault(); state.active = Math.min(items.length - 1, state.active + 1); paintActive(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); state.active = Math.max(0, state.active - 1); paintActive(); }
    else if (e.key === 'Enter') { e.preventDefault(); const li = items[Math.max(0, state.active)]; if (li) choose(li.dataset.id); }
  });
  document.addEventListener('click', e => { if (!e.target.closest('.combo')) closeList(); });
  try { state.directory = await api('/api/patients'); } catch (e) { toast(e.message); }
  renderQuick();
}
function renderQuick() {
  const recent = getRecent(), ex = examples.map(x => x.id);
  const chips = (ids, title) => ids.length ? `<span>${title}</span>${ids.map(id => `<button class="chip" data-id="${esc(id)}">${esc(id)}</button>`).join('')}` : '';
  $('#pt-quick').innerHTML = chips(recent, 'Recent:') + (recent.length && ex.length ? '<span style="width:12px"></span>' : '') + chips(ex.filter(x => !recent.includes(x)).slice(0, 4), 'Examples:');
  $$('#pt-quick .chip').forEach(b => b.onclick = () => choose(b.dataset.id));
}
function matches(q) {
  q = q.trim().toLowerCase().replace(/^vanco_?/, '');
  const list = state.directory;
  if (!q) return list.slice(0, 50);
  return list.filter(p => p.id.toLowerCase().includes(q)).slice(0, 50);
}
function openList() {
  const q = $('#pt-search').value, list = $('#pt-list'), rows = matches(q);
  const hl = id => { const k = q.trim().toLowerCase().replace(/^vanco_?/, ''); if (!k) return esc(id); const i = id.toLowerCase().lastIndexOf(k); return i < 0 ? esc(id) : esc(id.slice(0, i)) + '<mark>' + esc(id.slice(i, i + k.length)) + '</mark>' + esc(id.slice(i + k.length)); };
  list.innerHTML = rows.length ? rows.map((p, i) => `<li role="option" data-id="${esc(p.id)}" aria-selected="${i === state.active}"><b>${hl(p.id)}</b>
      <span class="meta">${p.age} y, ${p.sex === 'F' ? 'female' : 'male'}, ${fmt(p.weight_kg)} kg, CrCL ${p.crcl} mL/min</span>
      <span>${p.icu ? tag('neutral', 'ICU') : ''} ${p.courses > 1 ? tag('brand', `${p.courses} courses`) : ''}</span></li>`).join('')
    + (state.directory.length > rows.length && !q ? `<li class="none">Showing 50 of ${state.directory.length}. Type to narrow the list.</li>` : '')
    : `<li class="none">No patient ID contains “${esc(q)}”. IDs look like VANCO_00028.</li>`;
  list.hidden = false; $('.combo').setAttribute('aria-expanded', 'true');
  $$('#pt-list li[data-id]').forEach(li => li.onmousedown = e => { e.preventDefault(); choose(li.dataset.id); });
}
function paintActive() { $$('#pt-list li[data-id]').forEach((li, i) => { li.setAttribute('aria-selected', i === state.active); if (i === state.active) li.scrollIntoView({ block: 'nearest' }); }); }
function closeList() { const l = $('#pt-list'); if (l) l.hidden = true; $('.combo')?.setAttribute('aria-expanded', 'false'); }
function choose(id) { closeList(); $('#pt-search').value = id; $('#pt-search').blur(); go('patients', id); }

async function loadPatient(id, keepSelection = false) {
  $('#pt-empty').hidden = true;
  const view = $('#pt-view'); view.hidden = false;
  if (!keepSelection) view.innerHTML = '<p class="muted">Loading patient…</p>';
  try {
    const j = await api(`/api/patients/${encodeURIComponent(id)}`);
    pushRecent(id); renderQuick();
    $('#pt-search').value = id;
    showPatient(j, keepSelection ? state.selected : j.courses.length - 1);
  } catch (e) {
    view.innerHTML = `<div class="empty"><h2>Patient not found</h2><p class="muted">${esc(e.message)}. Check the ID or pick a patient from the list.</p></div>`;
  }
}

function currentEstimate(j) {
  const last = j.courses[j.courses.length - 1];
  return last.response ? { auc: last.response.auc24, cat: last.response.interpretation, basis: `Bayesian estimate from ${j.levels.length} levels` }
    : { auc: last.prediction.auc24, cat: last.prediction.interpretation, basis: last.prediction.basis_levels ? `Bayesian estimate from ${last.prediction.basis_levels} levels` : 'Covariate-only estimate' };
}
const courseAuc = c => c.response ? c.response.auc24 : c.prediction.auc24;
const courseCat = c => (c.response ? c.response.interpretation : c.prediction.interpretation).category;
const regLabel = r => `${fmt(r.dose_mg)} mg every ${fmt(r.interval_h, r.interval_h % 1 ? 1 : 0)} h`;
const hours = h => h < 48 ? `${fmt(h)} h` : `day ${fmt(Math.floor(h / 24) + 1)} (${fmt(h)} h)`;

function showPatient(j, sel) {
  state.patient = j; state.selected = Math.max(0, Math.min(sel, j.courses.length - 1));
  const p = j.patient, est = currentEstimate(j), n = j.courses.length;
  const first = j.courses[0], last = j.courses[n - 1];
  const doseChange = (last.regimen.daily_dose_mg / first.regimen.daily_dose_mg - 1) * 100;
  const onTarget = j.courses.filter(c => courseCat(c) === 'within').length;
  $('#pt-view').innerHTML = `
    <div class="pt-head">
      <div>
        <div class="pt-id">${esc(p.id)}</div>
        <div class="pt-facts"><span><b>${p.age}</b> years</span><span><b>${p.sex === 'F' ? 'Female' : 'Male'}</b></span><span><b>${fmt(p.weight_kg, 1)}</b> kg</span>
          <span>SCr <b>${fmt(p.SCr_mg_dl, 2)}</b> mg/dL</span><span>CrCL <b>${p.CrCL_ml_min}</b> mL/min</span><span>MIC <b>${p.MIC_mg_l}</b> mg/L</span></div>
        <div class="pt-tags">${p.ICU_flag ? tag('neutral', 'ICU') : ''}${p.ARC_flag ? tag('neutral', 'Augmented renal clearance') : ''}${p.RRT_flag ? tag('neutral', 'Renal replacement therapy') : ''}
          ${tag('neutral', p.subpopulation[0].toUpperCase() + p.subpopulation.slice(1) + ' group')}${p.split === 'test' ? tag('brand', 'Held-out test patient') : ''}</div>
      </div>
      <div class="pt-status">
        <span class="muted small">Current regimen: ${regLabel(last.regimen)}</span>
        <div class="big-auc">${fmt(est.auc)}<small>mg·h/L AUC24</small></div>
        ${tag(est.cat.category, est.cat.label)}
        <p>${est.basis}</p>
      </div>
    </div>

    <section class="ribbon-wrap" aria-label="Treatment journey">
      <div class="section-title"><h2>Treatment journey</h2><p>Each card is a course. Select one to see why it was given. Colours show the exposure category.</p></div>
      <div class="ribbon" role="group">${ribbon(j)}</div>
    </section>

    <div id="course-detail"></div>

    ${planner(j)}

    <section aria-label="Patient analytics">
      <div class="section-title"><h2>Patient analytics</h2><p>Latest individual estimate, projected 72 h into the recommended course.</p></div>
      <div class="kpis">
        <div class="kpi accent"><b>${fmt(est.auc)}</b><span>Current AUC24 estimate, mg·h/L</span></div>
        <div class="kpi"><b>${onTarget} of ${n}</b><span>Courses within the ${j.target[0]}–${j.target[1]} target</span></div>
        <div class="kpi"><b>${n > 1 ? sign(doseChange) + '%' : '—'}</b><span>Daily dose change since course 1</span></div>
        <div class="kpi"><b>${fmt(j.next.clearance_L_h, 2)} L/h</b><span>Clearance estimate, ±${fmt((Math.exp(1.2816 * j.next.log_cl_sd) - 1) * 100)}% (80%)</span></div>
      </div>
      <div class="grid-charts">
        <article class="panel span2"><div class="section-title"><h3>Concentration over time</h3>
          <span class="legend-inline"><span><i style="background:${C.ultra}"></i>Model curve</span><span><i style="background:${C.hebe}"></i>Projected next course</span><span><i style="background:${C.river};border-radius:50%"></i>Dataset level</span><span><i style="background:${C.violet};border-radius:50%"></i>Entered level</span><span><i style="border:2px solid ${C.river};background:#fff;border-radius:50%"></i>Simulated level</span></span></div>
          <div class="chart-box tall"><canvas id="ch-pt-time"></canvas></div>
          <form id="level-form" class="grid4" style="margin-top:14px;align-items:end" novalidate>
            <label><span class="lab">Level time <span class="unit">h after first dose</span></span><input name="time_h" type="number" step="any" min="0.1" required></label>
            <label><span class="lab">Concentration <span class="unit">mg/L</span></span><input name="conc_mg_L" type="number" step="any" min="0" max="200" required></label>
            <button class="btn secondary" type="submit">Add measured level</button>
            <span class="muted small">Levels update every estimate on this page.</span>
          </form>
        </article>
        <article class="panel"><h3>AUC24 by course</h3><p class="muted small">Predicted when the course was chosen, and re-estimated after its levels. Green band: target.</p><div class="chart-box"><canvas id="ch-pt-auc"></canvas></div></article>
        <article class="panel"><h3>Dose, peak and trough</h3><p class="muted small">Daily dose per course with predicted steady-state peak and trough.</p><div class="chart-box"><canvas id="ch-pt-dose"></canvas></div></article>
      </div>
    </section>

    <section aria-label="Full history">
      <div class="section-title"><h2>Full history</h2><p>Every course in order. The most recent added course can be removed.</p></div>
      <div class="panel table-wrap">${historyTable(j)}</div>
    </section>`;
  $$('.node[data-i]').forEach(b => b.onclick = () => selectCourse(+b.dataset.i));
  $('.node.next')?.addEventListener('click', () => $('#planner').scrollIntoView({ behavior: 'smooth' }));
  $$('[data-undo]').forEach(b => b.onclick = () => removeCourse(+b.dataset.undo));
  $('#level-form').onsubmit = addLevelSubmit;
  wirePlanner(j);
  selectCourse(state.selected, false);
  drawPatientCharts(j);
}

function ribbon(j) {
  const parts = [];
  j.courses.forEach((c, i) => {
    if (i) {
      const ch = (c.regimen.daily_dose_mg / j.courses[i - 1].regimen.daily_dose_mg - 1) * 100;
      parts.push(`<div class="link-arrow" aria-hidden="true"><span class="${ch > 0 ? 'up' : 'down'}">${Math.abs(ch) < 1 ? 'same dose' : sign(ch) + '%'}</span></div>`);
    }
    const cat = courseCat(c);
    parts.push(`<button class="node" data-i="${i}" aria-pressed="false" aria-label="Course ${c.index}: ${regLabel(c.regimen)}, AUC ${fmt(courseAuc(c))}">
      <div class="n-top"><span>Course ${c.index}, from ${fmt(c.start_h)} h</span><span class="dot" style="background:${C[cat]}"></span></div>
      <div class="n-dose">${regLabel(c.regimen)}</div>
      <div class="n-auc">AUC24 ${fmt(courseAuc(c))} ${c.response ? '(levels)' : '(predicted)'}</div>
      <div class="n-src">${{ dataset: 'Dataset starting regimen', recommended: 'VancoDose recommendation', clinician: 'Clinician choice' }[c.source]}</div></button>`);
  });
  const b = j.next.recommendation.best, last = j.courses[j.courses.length - 1];
  const ch = (b.daily_dose_mg / last.regimen.daily_dose_mg - 1) * 100;
  parts.push(`<div class="link-arrow" aria-hidden="true"><span class="${ch > 0 ? 'up' : 'down'}">${Math.abs(ch) < 1 ? 'same dose' : sign(ch) + '%'}</span></div>`);
  parts.push(`<button class="node next" aria-label="Recommended next course"><div class="n-top"><span>Next, from ${fmt(j.next.start_h)} h</span><span class="dot" style="background:${C.violet}"></span></div>
    <div class="n-dose">${b.label}</div><div class="n-auc">Predicted AUC24 ${fmt(b.auc24)}</div><div class="n-src">Recommended. Plan it below</div></button>`);
  return parts.join('');
}

function selectCourse(i, scroll = true) {
  const j = state.patient; state.selected = i;
  $$('.node[data-i]').forEach(b => b.setAttribute('aria-pressed', +b.dataset.i === i));
  const c = j.courses[i], nxt = j.courses[i + 1], cond = c.condition, pr = c.prediction, r = c.regimen;
  const next = nxt ? { label: regLabel(nxt.regimen), ch: (nxt.regimen.daily_dose_mg / r.daily_dose_mg - 1) * 100, from: nxt.start_h, src: nxt.source }
    : { label: j.next.recommendation.best.label, ch: (j.next.recommendation.best.daily_dose_mg / r.daily_dose_mg - 1) * 100, from: j.next.start_h, src: 'next' };
  const resp = c.response;
  const factors = c.explanation.factors.filter(f => Math.abs(f.dose_effect_pct) >= 3);
  const lvAdj = c.explanation.levels_adjustment;
  $('#course-detail').innerHTML = `
  <article class="course" aria-live="polite">
    <div class="course-head"><div><h2>Course ${c.index}: ${regLabel(r)}</h2>
      <span class="muted">From ${hours(c.start_h)}${c.end_h ? ` to ${hours(c.end_h)}` : ', ongoing'}</span></div>
      ${tag(courseCat(c), (resp ? resp.interpretation : pr.interpretation).label)}</div>
    <div class="flow">
      <div class="step"><h4><span>1</span>Condition</h4>
        <dl class="kv"><dt>Weight</dt><dd>${fmt(cond.weight_kg, 1)} kg</dd><dt>Creatinine</dt><dd>${fmt(cond.SCr_mg_dl, 2)} mg/dL</dd>
        <dt>CrCL</dt><dd>${fmt(cond.CrCL_ml_min)} mL/min</dd><dt>ICU</dt><dd>${cond.ICU_flag ? 'Yes' : 'No'}</dd>
        ${cond.ARC_flag ? '<dt>ARC</dt><dd>Yes</dd>' : ''}${cond.RRT_flag ? '<dt>RRT</dt><dd>Yes</dd>' : ''}</dl></div>
      <div class="step"><h4><span>2</span>Dose</h4>
        <div class="headline">${fmt(r.dose_mg)} mg</div>
        <div class="sub">every ${fmt(r.interval_h, 1)} h, over ${fmt(r.infusion_duration_h, 2)} h</div>
        <dl class="kv" style="margin-top:8px"><dt>Per kg</dt><dd>${fmt(r.mg_per_kg, 1)} mg/kg</dd><dt>Daily</dt><dd>${fmt(r.daily_dose_mg)} mg</dd>
        <dt>Predicted</dt><dd>${fmt(pr.auc24)}</dd><dt>In target</dt><dd>${fmt(pr.probability_in_target * 100)}%</dd></dl></div>
      <div class="step"><h4><span>3</span>Response</h4>
        ${resp ? `<div class="headline">${fmt(resp.auc24)} <span class="muted small">mg·h/L</span></div>
          <div class="sub">${tag(resp.interpretation.category, resp.interpretation.label)}</div>
          <div style="margin-top:8px">${resp.levels.map(l => `<div class="lv"><span>${fmt(l.time_h, 1)} h <span class="src">${l.source}</span></span><span><b>${fmt(l.conc, 1)}</b> <span class="muted">vs ${fmt(l.expected, 1)} expected</span></span></div>`).join('')}</div>`
          : `<div class="sub">No levels in this course.</div><div class="headline" style="margin-top:6px">${fmt(pr.auc24)} <span class="muted small">predicted</span></div>
             <p class="muted small" style="margin-top:6px">Add a level to see the observed response.</p>`}
        ${c.simulator_true_auc24 ? `<p class="muted small" style="margin-top:8px">Simulator truth (synthetic only): ${fmt(c.simulator_true_auc24)}</p>` : ''}</div>
      <div class="step"><h4><span>4</span>Reason</h4><ul>${c.reason.map(s => `<li>${esc(s)}</li>`).join('')}</ul></div>
      <div class="step"><h4><span>5</span>${nxt ? 'Next dose given' : 'Next dose'}</h4>
        <div class="headline">${next.label}</div>
        <div class="sub">${Math.abs(next.ch) < 1 ? 'Same daily dose' : `${sign(next.ch)}% daily dose`}, from ${fmt(next.from)} h</div>
        <p class="muted small" style="margin-top:8px">${nxt ? ({ recommended: 'Accepted VancoDose recommendation.', clinician: 'Clinician-chosen regimen.' }[next.src] || '') : 'Recommended now. See the planner below.'}</p>
        ${nxt ? `<button class="link" style="margin-top:6px" data-go="${i + 1}">Open course ${i + 2}</button>` : ''}</div>
    </div>
    ${c.flags.length ? `<div class="flags">${c.flags.map(f => `<div class="flag ${f.level}">${esc(f.text)}</div>`).join('')}</div>` : ''}
    <div class="course-xai">
      <div class="section-title"><h3>What drove the dose decision for course ${c.index}</h3>
        <span class="legend-inline"><span><i style="background:${C.violet}"></i>needs a higher dose</span><span><i style="background:${C.river}"></i>needs a lower dose</span></span></div>
      <p class="muted small">Each factor's effect on the dose needed to reach the target, from the estimate available when this course was chosen (${pr.basis_levels ? pr.basis_levels + ' levels' : 'no levels yet'}).</p>
      ${doseFactors(factors, lvAdj)}
    </div>
  </article>`;
  $$('#course-detail [data-go]').forEach(b => b.onclick = () => selectCourse(+b.dataset.go));
  if (scroll) $('#course-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function doseFactors(factors, lvAdj) {
  const all = factors.map(f => ({ label: f.label, value: f.value, e: f.dose_effect_pct }));
  if (lvAdj && Math.abs(lvAdj.dose_effect_pct) >= 1) all.push({ label: lvAdj.label, value: 'Bayesian update', e: lvAdj.dose_effect_pct });
  if (!all.length) return '<p class="muted">This patient is close to the typical patient; no single factor changes the dose by more than 3%.</p>';
  const max = Math.max(0.15, ...all.map(f => Math.abs(Math.log1p(f.e / 100))));
  return all.map(f => { const up = f.e > 0, w = Math.abs(Math.log1p(f.e / 100)) / max * 50;
    return `<div class="factor"><div class="name"><b>${esc(f.label)}</b><span>${esc(f.value)}</span></div><div class="bar"><i class="${up ? 'up' : 'down'}" style="width:${w}%"></i></div>
      <div class="pct" style="color:${up ? C.violet700 : '#1C8FCB'}">${sign(f.e)}%</div></div>`; }).join('');
}

function planner(j) {
  const n = j.next, b = n.recommendation.best, c = n.condition;
  const top = n.explanation.factors.filter(f => Math.abs(f.dose_effect_pct) >= 3).slice(0, 3);
  return `
  <section id="planner" class="planner" aria-label="Plan the next course">
    <div class="rec">
      <span class="card-label">Recommended next regimen</span>
      <div class="big">${fmt(b.dose_mg)} mg<small>every ${fmt(b.interval_h)} h</small></div>
      <div class="sub">Infuse over ${fmt(b.infusion_duration_h, 2)} h, ${fmt(b.daily_dose_mg)} mg/day, starting at ${fmt(n.start_h)} h</div>
      <div class="stats"><div class="stat"><b>${fmt(b.auc24)}</b><span>Predicted AUC24</span></div><div class="stat"><b>${fmt(b.probability_in_target * 100)}%</b><span>Chance in target</span></div><div class="stat"><b>${fmt(b.trough_ss, 1)}</b><span>Trough, mg/L</span></div></div>
      <ul>${n.reason.map(s => `<li>${esc(s)}</li>`).join('')}</ul>
      ${top.length ? `<div style="margin-top:16px"><span class="card-label">Biggest influences on this dose</span>${doseFactors(top, n.explanation.levels_adjustment)}</div>` : ''}
    </div>
    <form id="plan-form" novalidate>
      <h3>Plan the next course</h3>
      <div class="grid2">
        <label><span class="lab">Start <span class="unit">h after first dose</span></span><input name="start_h" type="number" step="any" value="${n.start_h}" required></label>
        <label><span class="lab">Dose <span class="unit">mg</span></span><input name="dose_mg" type="number" step="any" min="100" max="4000" value="${b.dose_mg}" required></label>
        <label><span class="lab">Every <span class="unit">hours</span></span><input name="interval_h" type="number" step="any" min="4" max="48" value="${b.interval_h}" required></label>
        <label><span class="lab">Infusion time <span class="unit">hours</span></span><input name="infusion_duration_h" type="number" step="any" min="0.5" max="24" value="${b.infusion_duration_h}" required></label>
      </div>
      <details><summary class="small" style="cursor:pointer;color:${C.violet700};font-weight:700">Update the patient's condition (weight, creatinine, ICU)</summary>
        <div class="grid2" style="margin-top:10px">
          <label><span class="lab">Weight <span class="unit">kg</span></span><input name="weight_kg" type="number" step="any" min="25" max="300" value="${c.weight_kg}"></label>
          <label><span class="lab">Serum creatinine <span class="unit">mg/dL</span></span><input name="scr" type="number" step="any" min="0.1" max="15" value="${c.SCr_mg_dl}"></label>
        </div>
        <div class="toggles"><label class="check"><input type="checkbox" name="icu" ${c.ICU_flag ? 'checked' : ''}> ICU</label><label class="check"><input type="checkbox" name="arc" ${c.ARC_flag ? 'checked' : ''}> Augmented renal clearance</label><label class="check"><input type="checkbox" name="rrt" ${c.RRT_flag ? 'checked' : ''}> Renal replacement therapy</label></div>
      </details>
      <label><span class="lab">Note <span class="unit">optional, saved with the course</span></span><textarea name="note" maxlength="500" placeholder="For example: creatinine rising, reduced dose as recommended"></textarea></label>
      <label class="check"><input type="checkbox" name="simulate" checked> Simulate a peak and trough for this course (synthetic patients only)</label>
      <div id="plan-preview" class="derived"></div>
      <div id="plan-error" class="form-error" role="alert" hidden></div>
      <div class="actions"><button class="btn primary" type="submit" id="plan-save">Start recommended regimen</button><button class="btn secondary" type="button" id="plan-reset" hidden>Reset to recommendation</button></div>
    </form>
  </section>`;
}

function wirePlanner(j) {
  const f = $('#plan-form'), b = j.next.recommendation.best, cond = j.next.condition;
  const isRec = () => +f.dose_mg.value === b.dose_mg && +f.interval_h.value === b.interval_h && +f.infusion_duration_h.value === b.infusion_duration_h;
  const condChanged = () => +f.weight_kg.value !== cond.weight_kg || +f.scr.value !== cond.SCr_mg_dl || f.icu.checked !== !!cond.ICU_flag || f.arc.checked !== !!cond.ARC_flag || f.rrt.checked !== !!cond.RRT_flag;
  const update = () => {
    const dose = +f.dose_mg.value, tau = +f.interval_h.value, rec = isRec();
    $('#plan-save').textContent = rec ? 'Start recommended regimen' : 'Save this regimen';
    $('#plan-reset').hidden = rec;
    if (dose > 0 && tau > 0) {
      const auc = dose * 24 / tau / j.next.clearance_L_h, cat = auc < j.target[0] ? 'below' : auc <= j.target[1] ? 'within' : auc <= 800 ? 'above' : 'toxic';
      $('#plan-preview').innerHTML = `${rec ? 'Recommended regimen' : 'Your regimen'}: ${fmt(dose * 24 / tau)} mg/day, predicted AUC24 <b>${fmt(auc)}</b> ${tag(cat)}`
        + (condChanged() ? ' <span class="changed">Condition changed: the estimate will be recalculated when you save.</span>' : '');
    } else $('#plan-preview').textContent = 'Enter a dose and an interval to see the predicted exposure.';
  };
  f.addEventListener('input', update); update();
  $('#plan-reset').onclick = () => { f.dose_mg.value = b.dose_mg; f.interval_h.value = b.interval_h; f.infusion_duration_h.value = b.infusion_duration_h; update(); };
  f.onsubmit = async e => {
    e.preventDefault(); const err = $('#plan-error'); err.hidden = true;
    if (!f.checkValidity()) { err.textContent = 'Check the highlighted fields: each value must be within its allowed range.'; err.hidden = false; f.reportValidity(); return; }
    if (+f.infusion_duration_h.value >= +f.interval_h.value) { err.textContent = 'Infusion time must be shorter than the dosing interval.'; err.hidden = false; return; }
    const btn = $('#plan-save'), label = btn.textContent; btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Saving';
    try {
      const body = { start_h: +f.start_h.value, accepted: isRec() && !condChanged(), simulate_levels: f.simulate.checked, note: f.note.value,
        regimen: { dose_mg: +f.dose_mg.value, interval_h: +f.interval_h.value, infusion_duration_h: +f.infusion_duration_h.value },
        condition: { weight_kg: +f.weight_kg.value, scr: +f.scr.value, icu: f.icu.checked ? 1 : 0, arc: f.arc.checked ? 1 : 0, rrt: f.rrt.checked ? 1 : 0 } };
      const nj = await api(`/api/patients/${encodeURIComponent(j.patient.id)}/courses`, { method: 'POST', body: JSON.stringify(body) });
      const added = nj.courses[nj.courses.length - 1];
      showPatient(nj, nj.courses.length - 1);
      toast(`Course ${added.index} saved: ${regLabel(added.regimen)}`, { label: 'Undo', run: () => removeCourse(added.id, true) });
    } catch (ex) { err.textContent = ex.message; err.hidden = false; btn.disabled = false; btn.textContent = label; }
  };
}

async function removeCourse(id, fromUndo = false) {
  const j = state.patient;
  if (!fromUndo && !confirm('Remove the most recent course and any levels simulated for it? This cannot be undone.')) return;
  try { const nj = await api(`/api/patients/${encodeURIComponent(j.patient.id)}/courses/${id}`, { method: 'DELETE' }); showPatient(nj, nj.courses.length - 1); toast('Course removed'); }
  catch (e) { toast(e.message); }
}

async function addLevelSubmit(e) {
  e.preventDefault(); const f = e.target, j = state.patient;
  if (!f.checkValidity()) { f.reportValidity(); return; }
  try {
    const nj = await api(`/api/patients/${encodeURIComponent(j.patient.id)}/levels`, { method: 'POST', body: JSON.stringify({ time_h: +f.time_h.value, conc_mg_L: +f.conc_mg_L.value }) });
    showPatient(nj, state.selected); toast(`Level ${fmt(+f.conc_mg_L.value, 1)} mg/L at ${fmt(+f.time_h.value, 1)} h added; estimates updated`);
  } catch (ex) { toast(ex.message); }
}

function historyTable(j) {
  const lastAdded = [...j.courses].reverse().find(c => c.id != null);
  return `<table><thead><tr><th>Course</th><th>Start</th><th>Regimen</th><th>mg/kg</th><th>mg/day</th><th>CrCL</th><th>Levels</th><th>Predicted AUC24</th><th>After levels</th><th>Status</th><th>Source</th><th></th></tr></thead><tbody>
    ${j.courses.map((c, i) => `<tr class="${i === state.selected ? 'sel' : ''}"><td><b>${c.index}</b></td><td>${fmt(c.start_h)} h</td><td>${regLabel(c.regimen)}</td><td>${fmt(c.regimen.mg_per_kg, 1)}</td>
      <td>${fmt(c.regimen.daily_dose_mg)}</td><td>${fmt(c.condition.CrCL_ml_min)}</td><td>${c.response ? c.response.levels.length : 0}</td><td>${fmt(c.prediction.auc24)}</td><td>${c.response ? fmt(c.response.auc24) : '—'}</td>
      <td>${tag(courseCat(c))}</td><td>${{ dataset: 'Dataset', recommended: 'Recommendation', clinician: 'Clinician' }[c.source]}</td>
      <td>${lastAdded && c.id === lastAdded.id ? `<button class="btn ghost" style="height:30px;padding:0 10px;font-size:12px" data-undo="${c.id}">Remove</button>` : ''}</td></tr>`).join('')}
  </tbody></table>`;
}

function drawPatientCharts(j) {
  const t = j.timeline, ps = t.projection_start_h;
  const solid = [], proj = [];
  t.t.forEach((x, i) => { const pt = { x, y: t.c[i] }; if (x <= ps) solid.push(pt); if (x >= ps) proj.push(pt); });
  const lv = src => j.levels.filter(l => l.source === src).map(l => ({ x: l.time_h, y: l.conc }));
  mkChart('ch-pt-time', { type: 'line', data: { datasets: [
      { label: 'Model curve', data: solid, borderColor: C.ultra, borderWidth: 1.8, pointRadius: 0 },
      { label: 'Projected next course', data: proj, borderColor: C.hebe, borderDash: [6, 4], borderWidth: 1.8, pointRadius: 0 },
      { type: 'scatter', label: 'Dataset level', data: lv('dataset'), backgroundColor: C.river, borderColor: '#fff', borderWidth: 2, pointRadius: 6 },
      { type: 'scatter', label: 'Entered level', data: lv('measured'), backgroundColor: C.violet, borderColor: '#fff', borderWidth: 2, pointRadius: 6 },
      { type: 'scatter', label: 'Simulated level', data: lv('simulated'), backgroundColor: '#fff', borderColor: C.river, borderWidth: 2.5, pointRadius: 5.5 }] },
    options: { ...baseOpts(), parsing: false, interaction: { mode: 'nearest', intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 1)} mg/L at ${fmt(c.parsed.x, 1)} h` } },
        vlines: { lines: [...j.courses.map(c => ({ x: c.start_h, label: `Course ${c.index}`, color: '#D6D1E8' })), { x: ps, label: 'Next', color: C.hebe, text: C.violet700 }] } },
      scales: { x: { type: 'linear', min: 0, max: t.t[t.t.length - 1], title: { display: true, text: 'Hours after first dose' }, grid: { color: '#F0EEF7' } },
                y: { min: 0, title: { display: true, text: 'Concentration (mg/L)' }, grid: { color: '#F0EEF7' } } } } });
  const labels = j.courses.map(c => `Course ${c.index}`).concat('Next');
  const b = j.next.recommendation.best;
  mkChart('ch-pt-auc', { type: 'bar', data: { labels, datasets: [
      { label: 'Predicted when chosen', data: j.courses.map(c => c.prediction.auc24).concat(b.auc24), backgroundColor: '#D9CCFF', borderRadius: 4 },
      { label: 'After measured levels', data: j.courses.map(c => c.response?.auc24 ?? null), backgroundColor: C.ultra, borderRadius: 4 }] },
    options: { ...baseOpts(), plugins: { ...baseOpts().plugins, band: { bands: [{ from: j.target[0], to: j.target[1], color: 'rgba(15,138,97,.12)', label: 'Target', text: C.within }, { from: 800, to: 803, color: 'rgba(207,51,67,.5)' }] } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#F0EEF7' }, title: { display: true, text: 'AUC24 (mg·h/L)' } } } } });
  mkChart('ch-pt-dose', { data: { labels, datasets: [
      { type: 'bar', label: 'Daily dose (mg)', data: j.courses.map(c => c.regimen.daily_dose_mg).concat(b.daily_dose_mg), backgroundColor: labels.map((_, i) => i === labels.length - 1 ? '#E3D6FF' : '#CDBBFF'), borderRadius: 4, yAxisID: 'y' },
      { type: 'line', label: 'Peak (mg/L)', data: j.courses.map(c => c.prediction.peak_ss).concat(b.peak_ss), borderColor: C.ultra, backgroundColor: C.ultra, yAxisID: 'y1', tension: 0.2 },
      { type: 'line', label: 'Trough (mg/L)', data: j.courses.map(c => c.prediction.trough_ss).concat(b.trough_ss), borderColor: C.river, backgroundColor: C.river, yAxisID: 'y1', tension: 0.2 }] },
    options: { ...baseOpts(), scales: { x: { grid: { display: false } }, y: { beginAtZero: true, position: 'left', title: { display: true, text: 'mg/day' }, grid: { color: '#F0EEF7' } },
      y1: { beginAtZero: true, position: 'right', title: { display: true, text: 'mg/L' }, grid: { display: false } } } } });
}

/* ================================================================ DOSE CALCULATOR */
const DEFAULT = { patient: { age: 58, sex: 'M', weight_kg: 82, SCr_mg_dl: 1.1, ICU_flag: 0, ARC_flag: 0, RRT_flag: 0 },
  regimen: { dose_mg: 1250, interval_h: 12, infusion_duration_h: 1.25, loading_dose_mg: 0 }, levels: [] };

async function initForm() {
  const f = $('#form');
  fill(DEFAULT);
  f.addEventListener('input', updateCrcl);
  $('#add-level').addEventListener('click', () => addLevel());
  ['t-lo', 't-hi'].forEach(id => $('#' + id).addEventListener('input', () => { $('#target-text').textContent = `${$('#t-lo').value}–${$('#t-hi').value} mg·h/L`; }));
  f.addEventListener('submit', e => { e.preventDefault(); calculate(); });
  try {
    examples = await api('/api/examples');
    const sel = $('#example');
    examples.forEach((x, i) => sel.insertAdjacentHTML('beforeend', `<option value="${i}">${x.label} (${x.id})</option>`));
    renderQuick();
    sel.addEventListener('change', () => { if (sel.value !== '') { fill(examples[+sel.value]); calculate(); } });
  } catch { /* examples are optional */ }
}

function fill(x) {
  const f = $('#form');
  for (const [k, v] of Object.entries({ ...x.patient, ...x.regimen })) {
    const el = f.elements[k]; if (!el) continue;
    if (el.type === 'checkbox') el.checked = !!+v; else el.value = v;
  }
  $('#levels').innerHTML = '';
  (x.levels || []).forEach(l => addLevel(l.time_h, l.conc_mg_L));
  updateCrcl();
}

function addLevel(t = '', c = '') {
  const row = document.createElement('div');
  row.className = 'level-row';
  row.innerHTML = `<label><span class="lab">Time <span class="unit">h after 1st dose</span></span><input type="number" step="any" min="0.1" max="72" value="${t}" data-k="time_h"></label>
    <label><span class="lab">Concentration <span class="unit">mg/L</span></span><input type="number" step="any" min="0" max="200" value="${c}" data-k="conc_mg_L"></label>
    <button type="button" aria-label="Remove level">×</button>`;
  row.querySelector('button').onclick = () => row.remove();
  $('#levels').appendChild(row);
}

function crclValue() {
  const f = $('#form').elements;
  const age = +f.age.value, w = +f.weight_kg.value, scr = +f.SCr_mg_dl.value;
  if (!age || !w || !scr) return null;
  return (140 - age) * w / (72 * scr) * (f.sex.value === 'F' ? 0.85 : 1);
}
function updateCrcl() {
  const v = crclValue();
  $('#crcl').innerHTML = v ? `Creatinine clearance (Cockcroft–Gault): <b>${fmt(v)} mL/min</b>` : 'Creatinine clearance: enter age, weight and creatinine';
}

function payload() {
  const f = $('#form').elements;
  const num = k => f[k].value === '' ? null : Number(f[k].value);
  const levels = [...document.querySelectorAll('.level-row')].map(r => ({ time_h: +r.querySelector('[data-k=time_h]').value, conc_mg_L: +r.querySelector('[data-k=conc_mg_L]').value }))
    .filter(l => l.time_h > 0 && r_ok(l.conc_mg_L));
  return {
    patient: { age: num('age'), sex: f.sex.value, weight_kg: num('weight_kg'), SCr_mg_dl: num('SCr_mg_dl'),
      ICU_flag: f.ICU_flag.checked ? 1 : 0, ARC_flag: f.ARC_flag.checked ? 1 : 0, RRT_flag: f.RRT_flag.checked ? 1 : 0 },
    regimen: { dose_mg: num('dose_mg'), interval_h: num('interval_h'), infusion_duration_h: num('infusion_duration_h'), loading_dose_mg: num('loading_dose_mg') || 0 },
    levels, target: [+$('#t-lo').value, +$('#t-hi').value],
  };
}
const r_ok = v => Number.isFinite(v) && v >= 0;

async function calculate() {
  const f = $('#form'), err = $('#form-error'), btn = $('#calc');
  err.hidden = true;
  if (!f.checkValidity()) { err.textContent = 'Please complete the highlighted fields with values in range.'; err.hidden = false; f.reportValidity(); return; }
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Calculating';
  try {
    const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload()) });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the inputs.');
    render(data);
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.textContent = 'Calculate exposure'; }
}

/* ---------------------------------------------------------------- results */
function gauge(auc, lo, hi, target) {
  const max = 1200, pos = v => Math.min(100, Math.max(0, v / max * 100));
  const [tl, th] = target;
  const seg = (a, b, col) => `<i style="width:${pos(b) - pos(a)}%;background:${col};opacity:.85"></i>`;
  return `<div class="gauge-wrap"><div class="gauge">${seg(0, tl, C.below)}${seg(tl, th, C.within)}${seg(th, 800, C.above)}${seg(800, max, C.toxic)}</div>
    <div class="marker-range" style="left:${pos(lo)}%;width:${Math.max(pos(hi) - pos(lo), 0.5)}%"></div>
    <div class="marker" style="left:${pos(auc)}%"></div>
    <div class="gauge-labels"><span style="left:0;transform:none">0</span><span style="left:${pos(tl)}%">${tl}</span><span style="left:${pos(th)}%">${th}</span><span style="left:${pos(800)}%">800</span><span style="right:0;left:auto;transform:none">${max}+</span></div></div>`;
}

function headline(r) {
  const c = r.interpretation.category, best = r.recommendation.best;
  const action = { below: 'Increase the dose', within: 'Continue the current regimen', above: 'Reduce the dose', toxic: 'Reduce the dose now' }[c];
  const sub = c === 'within' ? r.interpretation.message
    : `${r.interpretation.message} ${best ? `Suggested: <b>${best.label}</b>.` : ''}`;
  return `<div class="decision st-${c}"><span class="badge">${r.interpretation.label}</span><div><h2>${action}</h2><p>${sub}</p></div></div>`;
}

function render(r) {
  $('#empty').hidden = true;
  const out = $('#out'); out.hidden = false;
  const b = r.recommendation.best, ex = r.explanation;
  const modeText = r.mode === 'bayesian' ? `Bayesian estimate using ${r.levels.length} measured level${r.levels.length > 1 ? 's' : ''}` : 'Covariate-only estimate (no levels entered)';
  const change = r.recommendation.daily_dose_change_pct;
  out.innerHTML = `
    ${headline(r)}
    ${r.warnings.map(w => `<div class="warn">${w}</div>`).join('')}
    <div class="cards">
      <div class="panel">
        <div class="card-label">Predicted exposure for this regimen</div>
        <div class="big">${fmt(r.auc24)}<small>mg·h/L AUC24</small></div>
        <div class="sub">80% range ${fmt(r.auc24_interval_80[0])}–${fmt(r.auc24_interval_80[1])} · ${fmt(r.probability_in_target * 100)}% chance within target</div>
        ${gauge(r.auc24, ...r.auc24_interval_80, r.target)}
        <div class="stats">
          <div class="stat"><b>${fmt(r.trough_ss, 1)}</b><span>Trough, mg/L (steady state)</span></div>
          <div class="stat"><b>${fmt(r.peak_ss, 1)}</b><span>Peak, mg/L (steady state)</span></div>
          <div class="stat"><b>${fmt(r.half_life_h, 0)} h</b><span>Elimination half-life</span></div>
        </div>
        <p class="muted small" style="margin:10px 0 0">${modeText}. Regimen: ${fmt(r.regimen.dose_mg)} mg every ${fmt(r.regimen.interval_h, 1)} h (${fmt(r.regimen.daily_dose_mg)} mg/day).</p>
      </div>
      <div class="panel rec-card">
        <div class="card-label">Recommended regimen</div>
        ${b ? `<div class="big">${fmt(b.dose_mg)} mg<small>every ${fmt(b.interval_h)} h</small></div>
        <div class="sub">Infuse over ${fmt(b.infusion_duration_h, 2)} h · ${fmt(b.daily_dose_mg)} mg/day ${change != null && Math.abs(change) >= 1 ? `<span class="change">${change > 0 ? '+' : ''}${fmt(change)}% vs current</span>` : ''}</div>
        <div class="stats">
          <div class="stat"><b>${fmt(b.auc24)}</b><span>Expected AUC24</span></div>
          <div class="stat"><b>${fmt(b.probability_in_target * 100)}%</b><span>Chance within target</span></div>
          <div class="stat"><b>${fmt(b.trough_ss, 1)}</b><span>Expected trough, mg/L</span></div>
        </div>
        ${r.recommendation.loading_dose_mg ? `<p class="sub" style="margin:12px 0 0">Starting therapy? Consider a <b>${fmt(r.recommendation.loading_dose_mg)} mg</b> loading dose (25 mg/kg, max 3 g).</p>` : ''}
        <p class="muted small" style="margin:10px 0 0">Aims for AUC24 ≈ ${fmt(r.recommendation.goal_auc24)}; doses rounded to 250 mg. ${r.recommendation.basis}. Draw a peak and trough after 24–48 h to confirm.</p>`
        : `<p>${r.recommendation.note}</p>`}
      </div>
    </div>

    <article class="panel section">
      <div class="section-title"><h3>Why the model predicts this</h3>
        <span class="legend-inline"><span><i style="background:${C.violet}"></i>raises exposure</span><span><i style="background:${C.river}"></i>lowers exposure</span></span></div>
      <p class="muted">Start from a typical patient on this regimen, then apply each patient factor${ex.levels_adjustment ? ' and the measured levels' : ''}.</p>
      <div class="chain">
        <span class="node-s">Typical patient <b>${fmt(ex.typical_auc)}</b></span><span class="arr">→</span>
        <span class="node-s">With this patient's factors <b>${fmt(ex.prior_auc24)}</b></span>
        ${ex.levels_adjustment ? `<span class="arr">→</span><span class="node-s">After measured levels <b>${fmt(r.auc24)}</b></span>` : ''}
      </div>
      <div id="factors">${factorRows(ex)}</div>
      <p class="muted small" style="margin:12px 0 0">${sentence(ex)}</p>
    </article>

    <article class="panel section">
      <h3>Concentration over time</h3>
      <p class="muted">First 72 hours of the current regimen${b ? ' compared with the recommended one' : ''}${r.levels.length ? ', with your measured levels' : ''}.</p>
      <div class="chart-box"><canvas id="ch-curve"></canvas></div>
    </article>

    <details class="acc"><summary>All regimen options</summary>
      <div class="table-wrap"><table><thead><tr><th>Regimen</th><th>mg/day</th><th>Expected AUC24</th><th>Chance in target</th><th>Trough</th><th>Peak</th></tr></thead><tbody>
      ${r.recommendation.options.map((o, i) => `<tr class="${i === 0 ? 'best' : ''}"><td>${o.label}</td><td>${fmt(o.daily_dose_mg)}</td><td>${fmt(o.auc24)}</td><td>${fmt(o.probability_in_target * 100)}%</td><td>${fmt(o.trough_ss, 1)}</td><td>${fmt(o.peak_ss, 1)}</td></tr>`).join('')}
      </tbody></table></div></details>
    <details class="acc"><summary>Pharmacokinetic details</summary>
      <div class="table-wrap"><table><tbody>
        <tr><td>Clearance</td><td>${fmt(r.clearance_L_h, 2)} L/h</td></tr>
        <tr><td>Central / peripheral volume</td><td>${fmt(r.volumes_L.V1, 1)} L / ${fmt(r.volumes_L.V2, 1)} L</td></tr>
        <tr><td>Creatinine clearance used</td><td>${fmt(r.patient.CrCL_ml_min)} mL/min</td></tr>
        ${r.levels.map(l => `<tr><td>Level at ${fmt(l.time_h, 1)} h</td><td>measured ${fmt(l.conc_mg_L, 1)} mg/L · model fit ${fmt(l.fitted_mg_L, 1)} mg/L</td></tr>`).join('')}
        <tr><td>Method</td><td>${ex.method}; two-compartment PK model with Bayesian (MAP) update</td></tr>
      </tbody></table></div></details>`;
  drawCurve(r);
  out.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function factorRows(ex) {
  const rows = ex.factors.filter(f => Math.abs(f.effect_pct) >= 3);
  const all = [...rows];
  if (ex.levels_adjustment) all.push({ label: ex.levels_adjustment.label, value: 'Bayesian update', effect_pct: ex.levels_adjustment.effect_pct, levels: true });
  const maxAbs = Math.max(0.15, ...all.map(f => Math.abs(Math.log1p(f.effect_pct / 100))));
  return all.map(f => {
    const w = Math.abs(Math.log1p(f.effect_pct / 100)) / maxAbs * 50;
    const up = f.effect_pct > 0;
    return `<div class="factor"><div class="name"><b>${f.label}</b><span>${f.value}</span></div>
      <div class="bar"><i class="${up ? 'up' : 'down'}" style="width:${w}%"></i></div>
      <div class="pct" style="color:${up ? C.violet : '#1C8FCB'}">${up ? '+' : ''}${fmt(f.effect_pct)}%</div></div>`;
  }).join('') || '<p class="muted">This patient is close to the typical patient; no factor changes exposure by more than 3%.</p>';
}

function sentence(ex) {
  const top = ex.factors.filter(f => Math.abs(f.effect_pct) >= 5).slice(0, 2);
  const parts = top.map(f => `${f.label.toLowerCase()} (${f.value}) ${f.effect_pct > 0 ? 'raises' : 'lowers'} exposure by about ${fmt(Math.abs(f.effect_pct))}%`);
  let s = parts.length ? `Main drivers: ${parts.join('; ')}.` : 'No single factor dominates.';
  if (ex.levels_adjustment) s += ' ' + ex.levels_adjustment.text;
  return s + ' Effects multiply: higher clearance means lower exposure for the same dose.';
}

function drawCurve(r) {
  if (!window.Chart) return;
  charts.curve?.destroy();
  const ds = [{ label: 'Current regimen', data: r.curve.map(p => ({ x: p.t, y: p.c })), borderColor: C.ultra, borderWidth: 2, pointRadius: 0, tension: 0 }];
  const b = r.recommendation.best;
  if (b && r.rec_curve) ds.push({ label: 'Recommended', data: r.rec_curve.map(p => ({ x: p.t, y: p.c })), borderColor: C.purple, borderDash: [6, 4], borderWidth: 2, pointRadius: 0 });
  if (r.levels.length) ds.push({ type: 'scatter', label: 'Measured levels', data: r.levels.map(l => ({ x: l.time_h, y: l.conc_mg_L })), backgroundColor: C.river, borderColor: '#fff', borderWidth: 2, pointRadius: 7 });
  charts.curve = new Chart($('#ch-curve'), { type: 'line', data: { datasets: ds },
    options: { animation: false, parsing: false, maintainAspectRatio: false, interaction: { mode: 'nearest', intersect: false },
      scales: { x: { type: 'linear', min: 0, max: 72, title: { display: true, text: 'Hours after first dose' }, ticks: { stepSize: 12 }, grid: { color: C.line } },
                y: { min: 0, title: { display: true, text: 'Concentration (mg/L)' }, grid: { color: C.line } } },
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 1)} mg/L at ${fmt(c.parsed.x, 1)} h` } } } } });
}

/* ================================================================ ANALYTICS */
async function loadAnalytics() {
  analyticsLoaded = true;
  const a = await api('/api/analytics');
  $('#an-note').textContent = `${a.n_patients} simulated patients. ${a.regimen_note}`;
  const o = a.overall, t = a.model_test?.target_attainment;
  $('#an-kpis').innerHTML = `
    <div class="kpi"><b>${fmt(o.within)}%</b><span>within 400–600 with standard dosing</span></div>
    <div class="kpi"><b>${fmt(o.toxic)}%</b><span>above 800 mg·h/L (toxicity concern)</span></div>
    <div class="kpi"><b>${fmt(o.median_auc)}</b><span>median AUC24, mg·h/L</span></div>
    ${t ? `<div class="kpi accent"><b>${fmt(t.recommended_after_levels.pct_within_400_600)}%</b><span>within target after model-guided dosing with 2 levels (test set)</span></div>` : ''}`;
  const catColor = l => { const lo = parseFloat(l); return l.startsWith('<') ? C.below : lo < 400 ? C.below : lo < 600 ? C.within : lo < 800 ? C.above : C.toxic; };
  charts.hist = new Chart($('#ch-hist'), { type: 'bar', data: { labels: a.auc_histogram.labels, datasets: [{ data: a.auc_histogram.counts, backgroundColor: a.auc_histogram.labels.map(catColor), borderRadius: 4 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, title: { display: true, text: 'AUC24 (mg·h/L)' } }, y: { grid: { color: C.line }, title: { display: true, text: 'Patients' } } } } });
  const g = a.subgroups;
  const stack = (k, col, lab) => ({ label: lab, data: g.map(x => x[k]), backgroundColor: col, borderWidth: 0 });
  charts.groups = new Chart($('#ch-groups'), { type: 'bar', data: { labels: g.map(x => `${x.group} (n=${x.n})`),
    datasets: [stack('below', C.below, 'Below'), stack('within', C.within, 'Within'), stack('above', C.above, 'Above'), stack('toxic', C.toxic, '>800')] },
    options: { indexAxis: 'y', plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.x, 1)}%` } } },
      scales: { x: { stacked: true, max: 100, grid: { color: C.line }, ticks: { callback: v => v + '%' } }, y: { stacked: true, grid: { display: false } } } } });
  $('#pt-note').textContent = `Each dot is a patient, coloured by true AUC24 category. At steady state, trough and AUC24 move together closely in this simulation (log-scale r = ${a.trough_auc_correlation}).`;
  const by = c => a.peak_trough.filter(p => p.cat === c).map(p => ({ x: p.trough, y: p.peak }));
  charts.pt = new Chart($('#ch-pt'), { type: 'scatter', data: { datasets: ['below', 'within', 'above', 'toxic'].map(c => ({ label: CAT[c], data: by(c), backgroundColor: C[c] + 'B0', pointRadius: 3.5 })) },
    options: { plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } } },
      scales: { x: { type: 'logarithmic', title: { display: true, text: 'Trough (mg/L)' }, grid: { color: C.line } }, y: { type: 'logarithmic', title: { display: true, text: 'Peak (mg/L)' }, grid: { color: C.line } } } } });
  if (t) {
    const keys = [['standard_dosing', 'Standard 15–20 mg/kg'], ['recommended_a_priori', 'Model, covariates only'], ['recommended_after_levels', 'Model + 2 levels']];
    const ds = (k, col, lab) => ({ label: lab, data: keys.map(([s]) => k === 'above' ? t[s].pct_above_600 : k === 'below' ? t[s].pct_below_400 : t[s].pct_within_400_600), backgroundColor: col });
    charts.strat = new Chart($('#ch-strat'), { type: 'bar', data: { labels: keys.map(k => k[1]), datasets: [ds('below', C.below, 'Below 400'), ds('within', C.within, 'Within 400–600'), ds('above', C.above, 'Above 600')] },
      options: { plugins: { legend: { position: 'bottom', labels: { boxWidth: 10 } }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 1)}%` } } },
        scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, max: 100, grid: { color: C.line }, ticks: { callback: v => v + '%' } } } } });
  }
  $('#an-risk').innerHTML = `<div class="table-wrap"><table><thead><tr><th>Group</th><th>Patients</th><th>Median AUC24</th><th>Below</th><th>Within</th><th>Above</th><th>&gt;800</th></tr></thead><tbody>
    ${a.renal_risk.map(r => `<tr><td>${r.group}</td><td>${r.n}</td><td>${fmt(r.median_auc)}</td><td>${fmt(r.below)}%</td><td>${fmt(r.within)}%</td><td>${fmt(r.above)}%</td><td><b style="color:${r.toxic > 40 ? C.toxic : C.ink}">${fmt(r.toxic)}%</b></td></tr>`).join('')}
    </tbody></table></div><p class="muted small">Standard weight-based dosing ignores kidney function, so patients with low creatinine clearance are systematically over-exposed and those with high clearance under-exposed.</p>`;
}

/* ---------------------------------------------------------------- model page */
async function loadModel() {
  modelLoaded = true;
  const m = await api('/api/model');
  const tm = m.card.test_metrics || {};
  const k = (x, d = 0) => x == null ? '—' : fmt(x, d);
  const ap = tm.a_priori_given_regimens || {}, bp = tm.bayes_peak_trough_given_regimens || {}, bt = tm.bayes_peak_trough_target_range_regimens || {};
  const tbl = (rows, cols, sel) => `<div class="table-wrap"><table><thead><tr>${cols.map(c => `<th>${c[1]}</th>`).join('')}</tr></thead><tbody>
    ${rows.map(r => `<tr class="${sel && sel(r) ? 'sel' : ''}">${cols.map(c => `<td>${typeof r[c[0]] === 'number' ? fmt(r[c[0]], c[2] ?? 1) : (r[c[0]] ?? '—')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const selName = m.card.selection?.apriori_model;
  $('#model-body').innerHTML = `
    <div class="kpis">
      <div class="kpi"><b>${k(ap.MAE)}</b><span>AUC24 MAE, covariates only (RMSE ${k(ap.RMSE)})</span></div>
      <div class="kpi accent"><b>${k(bp.MAE)}</b><span>AUC24 MAE with peak + trough (RMSE ${k(bp.RMSE)})</span></div>
      <div class="kpi accent"><b>${k(bt.MAE)}</b><span>MAE on target-range regimens (RMSE ${k(bt.RMSE)})</span></div>
      <div class="kpi"><b>${k((tm.bayes_80pct_interval_coverage || 0) * 100)}%</b><span>of true AUCs inside the 80% range</span></div>
    </div>
    ${goalCallout(tm, m.goal)}
    <div class="pipeline">
      <div class="pstep"><h3>Covariate model</h3><p>${selName ? selName[0].toUpperCase() + selName.slice(1).replace('_', ' ') : 'Linear'} regression on log-clearance using age, weight, Cockcroft–Gault CrCL, ICU, RRT and ARC. Chosen from 8 algorithms tuned with Optuna.</p></div>
      <div class="pstep"><h3>Bayesian update</h3><p>If levels are entered, a two-compartment PK model combines them with the covariate estimate (maximum a-posteriori), as recommended by the 2020 vancomycin consensus guideline.</p></div>
      <div class="pstep"><h3>Decision</h3><p>AUC24 = daily dose ÷ clearance for any regimen, so the same estimate ranks practical regimens (250 mg steps, q8/q12/q24/q48h) by their chance of reaching target.</p></div>
    </div>
    <article class="panel section"><h3>Covariate-only algorithms</h3><p class="muted">5-fold patient-grouped cross-validation × 3 repeats on ${m.card.training_patients - 80} development patients. Highlighted: selected model.</p>
      ${tbl(m.apriori, [['model', 'Model'], ['CV_MAE', 'MAE'], ['CV_RMSE', 'RMSE'], ['CV_R2', 'R²', 2], ['MAE_fold_SD', 'Fold SD'], ['overfit_ratio_RMSE', 'CV/train RMSE', 2], ['inference_us_per_patient', 'µs / patient', 1], ['explainability', 'Explanations']], r => r.model === selName)}</article>
    <article class="panel section"><h3>Using measured levels</h3><p class="muted">Same cross-validation. Levels are taken from the noisy simulated assay at protocol times (trough before dose 4; peak 1 h after infusion 3).</p>
      ${tbl(m.tdm, [['design', 'Levels'], ['model', 'Method'], ['CV_MAE', 'MAE'], ['CV_RMSE', 'RMSE'], ['CV_R2', 'R²', 2], ['CV_MdAPE_%', 'Median % error', 1], ['category_agreement', 'Category agreement', 2]], r => r.model.startsWith('Bayesian'))}</article>
    <article class="panel section"><h3>Concentration-time prediction</h3><p class="muted">Hourly concentrations over 72 h versus the noise-free simulated truth.</p>
      ${tbl(m.curves, [['model', 'Method'], ['MAE_vs_true_mg_L', 'MAE (mg/L)', 2], ['RMSE_vs_true_mg_L', 'RMSE (mg/L)', 2], ['R2_vs_true', 'R²', 3]], r => r.model.startsWith('Mechanistic, Bayesian'))}</article>
    <article class="panel section"><h3>Held-out test error by exposure band</h3><p class="muted">80 patients never used for development. Error grows with exposure.</p>
      ${tbl(m.bands, [['model', 'Model'], ['true AUC band', 'True AUC24'], ['n', 'n', 0], ['MAE', 'MAE'], ['RMSE', 'RMSE'], ['MdAPE_%', 'Median % error', 1]])}</article>
    <article class="panel section"><h3>Data handling &amp; limitations</h3>
      <ul class="sub">
        <li>Excluded as inputs (leakage): all latent "true_" PK parameters, simulator seed, AUC/AUC-MIC/trapezoidal AUC, steady-state peak/trough labels, noise-free concentrations, and the dose of the "auc_targeted" regimens (designed using the latent clearance).</li>
        <li>Patients are never split across training and evaluation. The 80 test patients were used once, after all choices were frozen.</li>
        <li>Training data are simulated (${m.card.training_patients} patients after excluding 6 outliers with serum creatinine &gt; 3.5 mg/dL; only 9 on renal replacement therapy). Predictions for SCr &gt; 3.5 mg/dL are outside the model's scope. Real-world validation is required before any clinical use.</li>
        <li>AUC24 assumes MIC = 1 mg/L, as in the 2020 ASHP/IDSA/PIDS/SIDP consensus guideline.</li>
      </ul></article>`;
}

function goalCallout(tm, goal = { MAE: 90, RMSE: 100 }) {
  const rows = [['Covariates only, all regimens', tm.a_priori_given_regimens],
    ['Peak + trough, all regimens', tm.bayes_peak_trough_given_regimens],
    ['Peak + trough, regimens aimed at the target range', tm.bayes_peak_trough_target_range_regimens],
    ['Two peak–trough pairs, regimens aimed at the target range', tm.bayes_two_pairs_target_range_regimens],
    ['Peak + trough, the regimen VancoDose recommends', tm.target_attainment?.recommended_after_levels_prediction_error]].filter(r => r[1]);
  const st = r => r.MAE < goal.MAE && r.RMSE < goal.RMSE ? '<span class="tag within">MAE and RMSE met</span>'
    : r.MAE < goal.MAE ? '<span class="tag below">MAE met, RMSE not</span>' : '<span class="tag above">Not met</span>';
  const any = rows.some(([, r]) => r.MAE < goal.MAE && r.RMSE < goal.RMSE);
  return `<div class="callout"><h3>Accuracy goal (MAE below ${goal.MAE}, RMSE below ${goal.RMSE} mg·h/L): ${any ? 'met for the regimens VancoDose recommends' : 'not met'}</h3>
    <p>AUC24 equals daily dose ÷ clearance exactly in this dataset, so accuracy depends only on how well clearance is estimated. About a quarter of the variation in clearance between patients is random, which is why every algorithm reaches the same covariate-only error. Measured levels remove much of it, and errors scale with exposure, so they are smallest for regimens near the target range.</p>
    <div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Held-out test, 80 patients</th><th>MAE</th><th>RMSE</th><th>Goal</th></tr></thead><tbody>
    ${rows.map(([n, r]) => `<tr><td>${n}</td><td>${fmt(r.MAE)}</td><td>${fmt(r.RMSE)}</td><td>${st(r)}</td></tr>`).join('')}
    </tbody></table></div></div>`;
}
