/* Dashboard and evaluation UI.
   Plain fetch + DOM, no framework. Five views swapped by a hash router. */

const $  = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const fmt = (n, d = 3) => (n ?? 0).toFixed(d);

const state = {
  wsId: sessionStorage.getItem('wsId') || null,
  files: [],
  questions: [],
  configs: [],
  limits: null,
  jobId: null,
  poll: null,
  runs: [],
  chart: null,
};

const CONFIG_PRESETS = [
  { name: 'small',  chunk_size: 400,  chunk_overlap: 50,  top_k: 5, desc: 'Precise retrieval, less surrounding context' },
  { name: 'medium', chunk_size: 800,  chunk_overlap: 100, top_k: 5, desc: 'Balanced. A sensible default to start from' },
  { name: 'large',  chunk_size: 1200, chunk_overlap: 150, top_k: 5, desc: 'More context per hit, coarser retrieval' },
];

const SAMPLES = {
  'hr_policy.txt':
    'Employees are entitled to 24 days of paid leave per calendar year.\n' +
    'Leave requests must be submitted at least 7 days in advance through the HR portal.\n' +
    'Unused leave may be carried forward up to a maximum of 10 days into the next year.\n' +
    'Sick leave is separate and capped at 12 days annually, requiring a medical certificate beyond 2 consecutive days.\n' +
    'Maternity leave is 26 weeks and paternity leave is 15 days, both fully paid.\n',
  'it_support.txt':
    'Laptops are refreshed every 3 years. Raise a ticket on the IT helpdesk for hardware issues.\n' +
    'VPN access requires multi-factor authentication set up through the company authenticator app.\n' +
    'Password resets can be self-served at reset.internal.com and passwords expire every 90 days.\n' +
    'Software installation requires manager approval for any tool not on the pre-approved list.\n' +
    'Standard response time for a P1 incident is 2 hours and P3 is 3 business days.\n',
  'benefits.txt':
    'Health insurance covers the employee, spouse, and up to two children with a sum insured of 5 lakh.\n' +
    'The company matches provident fund contributions at 12 percent of basic salary.\n' +
    'An annual wellness allowance of 15000 rupees can be claimed against gym or fitness expenses.\n' +
    'Employees become eligible for the performance bonus after completing 6 months of service.\n' +
    'Referral bonus is 25000 rupees paid after the referred candidate completes 90 days.\n',
};

const SAMPLE_QUESTIONS = [
  { question: 'How many days of paid leave do employees get per year?', expected_answer: '24 days per calendar year.', doc: 'hr_policy.txt' },
  { question: 'What is the response time for a P1 incident?', expected_answer: '2 hours.', doc: 'it_support.txt' },
  { question: 'How much is the referral bonus and when is it paid?', expected_answer: '25000 rupees, paid after the referred candidate completes 90 days.', doc: 'benefits.txt' },
];

/* ------------------------------------------------------------------ api */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep status */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const apiJson = (path, method, body) =>
  api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function banner(msg, kind = 'err') {
  $('#banner').innerHTML = msg ? `<div class="banner banner-${kind}">${msg}</div>` : '';
  if (msg) window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* --------------------------------------------------------------- router */

const VIEWS = ['home', 'new', 'run', 'results', 'history'];

function route() {
  const hash = (location.hash || '#/').replace('#/', '') || 'home';
  const name = VIEWS.includes(hash) ? hash : 'home';

  VIEWS.forEach(v => $(`#view-${v}`).classList.toggle('hidden', v !== name));
  $$('.nav-links a[data-nav]').forEach(a =>
    a.classList.toggle('active', a.dataset.nav === name));
  banner('');

  if (name === 'new') initNew();
  if (name === 'history') loadHistory();
}

window.addEventListener('hashchange', route);

/* ------------------------------------------------------- new: workspace */

async function ensureWorkspace() {
  if (state.wsId) {
    try {
      const ws = await api(`/workspace/${state.wsId}`);
      state.files = ws.files;
      return ws;
    } catch {
      state.wsId = null;            // expired — fall through and make a new one
      sessionStorage.removeItem('wsId');
    }
  }
  const ws = await api('/workspace', { method: 'POST' });
  state.wsId = ws.id;
  state.files = [];
  sessionStorage.setItem('wsId', ws.id);
  return ws;
}

let newReady = false;

async function initNew() {
  if (newReady) { renderFiles(); renderQuestions(); return; }

  try {
    state.limits = await api('/workspace/limits');
    const L = state.limits;
    $('#limit-note').textContent =
      `up to ${L.max_files} files, ${L.max_file_mb} MB each`;
    $('#drop-hint').textContent =
      L.allowed_extensions.join(', ').replaceAll('.', '').toUpperCase();
    if (!L.free_tier_available) {
      banner('This server has no API keys configured. Add your own keys below to run an evaluation.', 'warn');
    }
  } catch (e) {
    banner(`Could not reach the server: ${e.message}`);
    return;
  }

  renderConfigs();
  try {
    const ws = await ensureWorkspace();
    updateRunNote(ws);
  } catch (e) {
    banner(e.message);
  }

  renderFiles();
  if (!state.questions.length) addQuestion();
  newReady = true;
}

function updateRunNote(ws) {
  $('#run-note').textContent =
    ws.runs_remaining > 0
      ? `${ws.runs_remaining} free run${ws.runs_remaining === 1 ? '' : 's'} left today`
      : 'No free runs left — add your own API keys above';
}

/* ----------------------------------------------------------- new: files */

function renderFiles() {
  const list = $('#file-list');
  if (!state.files.length) { list.innerHTML = ''; return; }

  list.innerHTML = state.files.map(f => `
    <div class="file-row">
      <span class="file-name">${f.name}</span>
      <span class="file-size">${fmtSize(f.size_bytes)}</span>
      <button class="btn btn-danger btn-sm" data-del="${f.name}">Remove</button>
    </div>`).join('');

  $$('[data-del]', list).forEach(b =>
    b.onclick = () => deleteFile(b.dataset.del));

  renderQuestions();   // doc dropdowns depend on the file list
  markSteps();
}

async function uploadFiles(fileList) {
  if (!fileList.length) return;
  try {
    await ensureWorkspace();
    const fd = new FormData();
    [...fileList].forEach(f => fd.append('files', f));
    const res = await api(`/workspace/${state.wsId}/files`, { method: 'POST', body: fd });
    state.files = res.workspace.files;
    renderFiles();
    banner('');
  } catch (e) {
    banner(e.message);
  }
}

async function deleteFile(name) {
  try {
    const ws = await api(`/workspace/${state.wsId}/files/${encodeURIComponent(name)}`, { method: 'DELETE' });
    state.files = ws.files;
    renderFiles();
  } catch (e) {
    banner(e.message);
  }
}

async function loadSamples() {
  const files = Object.entries(SAMPLES).map(([name, text]) =>
    new File([text], name, { type: 'text/plain' }));
  await uploadFiles(files);

  if (state.questions.every(q => !q.question)) {
    state.questions = SAMPLE_QUESTIONS.map(q => ({
      question: q.question,
      expected_answer: q.expected_answer,
      relevant_doc_ids: [q.doc],
    }));
    renderQuestions();
  }
}

/* ------------------------------------------------------- new: questions */

function addQuestion() {
  state.questions.push({ question: '', expected_answer: '', relevant_doc_ids: [] });
  renderQuestions();
}

function renderQuestions() {
  const list = $('#q-list');
  const opts = state.files.map(f => f.name);

  list.innerHTML = state.questions.map((q, i) => `
    <div class="q-row">
      <div class="q-row-head">
        <span class="q-num">Question ${i + 1}</span>
        ${state.questions.length > 1
          ? `<button class="btn btn-danger btn-sm" data-qdel="${i}">Remove</button>` : ''}
      </div>
      <div class="field">
        <label>Question</label>
        <input type="text" data-q="${i}" value="${esc(q.question)}"
               placeholder="What does the policy say about...?">
      </div>
      <div class="field">
        <label>Expected answer</label>
        <textarea data-a="${i}" placeholder="What a correct answer should contain">${esc(q.expected_answer)}</textarea>
      </div>
      <div class="field" style="margin-bottom:0">
        <label>Which document has the answer?</label>
        <select data-d="${i}">
          <option value="">Any document</option>
          ${opts.map(o => `<option value="${o}" ${q.relevant_doc_ids[0] === o ? 'selected' : ''}>${o}</option>`).join('')}
        </select>
      </div>
    </div>`).join('');

  $$('[data-q]', list).forEach(el =>
    el.oninput = () => { state.questions[el.dataset.q].question = el.value; markSteps(); });
  $$('[data-a]', list).forEach(el =>
    el.oninput = () => { state.questions[el.dataset.a].expected_answer = el.value; markSteps(); });
  $$('[data-d]', list).forEach(el =>
    el.onchange = () => {
      state.questions[el.dataset.d].relevant_doc_ids = el.value ? [el.value] : [];
    });
  $$('[data-qdel]', list).forEach(b =>
    b.onclick = () => { state.questions.splice(+b.dataset.qdel, 1); renderQuestions(); });

  markSteps();
}

const esc = s => (s || '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const fmtSize = b =>
  b < 1024 ? `${b} B`
  : b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB`
  : `${(b / 1024 / 1024).toFixed(1)} MB`;

/* --------------------------------------------------------- new: configs */

function renderConfigs() {
  state.configs = ['small', 'large'];   // sensible default comparison
  $('#cfg-grid').innerHTML = CONFIG_PRESETS.map(c => `
    <div class="cfg ${state.configs.includes(c.name) ? 'on' : ''}" data-cfg="${c.name}">
      <div class="cfg-name">${c.name}</div>
      <div class="cfg-desc">${c.desc}</div>
      <div class="cfg-desc mono" style="margin-top:6px">chunk ${c.chunk_size} · k=${c.top_k}</div>
    </div>`).join('');

  $$('[data-cfg]').forEach(el => el.onclick = () => {
    const name = el.dataset.cfg;
    const i = state.configs.indexOf(name);
    if (i >= 0) {
      if (state.configs.length === 1) return;     // keep at least one
      state.configs.splice(i, 1);
    } else {
      const max = state.limits?.max_configs_per_run ?? 3;
      if (state.configs.length >= max) {
        banner(`You can compare at most ${max} configurations per run.`, 'warn');
        return;
      }
      state.configs.push(name);
    }
    el.classList.toggle('on');
    banner('');
  });
}

function markSteps() {
  const ok1 = state.files.length > 0;
  const ok2 = state.questions.some(q => q.question.trim() && q.expected_answer.trim());
  const set = (n, done, active) => {
    const el = $(`.step[data-step="${n}"]`);
    el.classList.toggle('ok', done);
    el.classList.toggle('on', active && !done);
  };
  set(1, ok1, true);
  set(2, ok1 && ok2, ok1);
  set(3, ok1 && ok2, ok1 && ok2);
}

/* ------------------------------------------------------------- new: run */

async function startRun() {
  const valid = state.questions
    .filter(q => q.question.trim() && q.expected_answer.trim())
    .map((q, i) => ({
      id: `q${i + 1}`,
      question: q.question.trim(),
      expected_answer: q.expected_answer.trim(),
      relevant_doc_ids: q.relevant_doc_ids,
    }));

  if (!state.files.length) return banner('Upload at least one document first.');
  if (!valid.length) return banner('Add at least one question with an expected answer.');

  const btn = $('#btn-run');
  btn.disabled = true;
  btn.textContent = 'Starting…';

  try {
    await apiJson(`/workspace/${state.wsId}/questions`, 'PUT', valid);

    const body = {
      configs: CONFIG_PRESETS.filter(c => state.configs.includes(c.name))
        .map(({ name, chunk_size, chunk_overlap, top_k }) => ({ name, chunk_size, chunk_overlap, top_k })),
      use_judge: $('#use-judge').checked,
    };
    const gk = $('#key-gemini').value.trim();
    const qk = $('#key-groq').value.trim();
    if (gk || qk) { body.gemini_key = gk; body.groq_key = qk; }

    const job = await apiJson(`/workspace/${state.wsId}/evaluate`, 'POST', body);
    state.jobId = job.job_id;

    location.hash = '#/run';
    $('#run-title').textContent = 'Running evaluation';
    $('#run-spin').classList.remove('hidden');
    pollProgress();
  } catch (e) {
    banner(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run evaluation';
  }
}

function pollProgress() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let p;
    try {
      p = await api(`/workspace/${state.wsId}/progress/${state.jobId}`);
    } catch (e) {
      clearInterval(state.poll);
      $('#run-spin').classList.add('hidden');
      banner(`Lost contact with the run: ${e.message}`);
      return;
    }

    $('#run-bar').style.width = `${p.percent}%`;
    $('#run-pct').textContent = `${Math.round(p.percent)}%`;
    $('#run-msg').textContent = p.message || p.phase;
    $('#run-eta').textContent = p.eta_s ? `~${Math.round(p.eta_s)}s left` : '';

    if (p.current_question) {
      $('#run-q').classList.remove('hidden');
      $('#run-q').textContent = p.current_question;
    }

    if (p.phase === 'done') {
      clearInterval(state.poll);
      $('#run-spin').classList.add('hidden');
      showResults(p.config_name.split(',').filter(Boolean));
    } else if (p.phase === 'error') {
      clearInterval(state.poll);
      $('#run-spin').classList.add('hidden');
      $('#run-title').textContent = 'Evaluation failed';
      banner(p.error || 'The evaluation failed.');
    }
  }, 1200);
}

/* --------------------------------------------------------------- results */

async function showResults(runIds) {
  location.hash = '#/results';
  const out = $('#results-body');
  out.innerHTML = '<div class="empty">Loading results…</div>';

  try {
    const runs = await Promise.all(runIds.map(id => api(`/runs/${id}`)));

    // Rank by MRR, then precision@k as a tiebreak.
    const best = [...runs].sort((a, b) =>
      (b.mrr - a.mrr) || (b.precision_at_k - a.precision_at_k))[0];

    const multi = runs.length > 1;

    const cards = runs.map(r => `
      <div class="kpi ${r.run_id === best.run_id && multi ? 'winner' : ''}">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="kpi-label">${r.config_name}</span>
          ${r.run_id === best.run_id && multi ? '<span class="badge">best</span>' : ''}
        </div>
        <div class="kpi-value">${fmt(r.mrr)}</div>
        <div class="small dim mono" style="margin-top:8px">
          hit ${fmt(r.hit_rate)} · p@k ${fmt(r.precision_at_k)}<br>
          faith ${fmt(r.avg_faithfulness, 2)} · ${Math.round(r.p50_latency_ms)}ms
        </div>
      </div>`).join('');

    // Config tabs, defaulting to the winner rather than the last run.
    const tabs = multi ? `
      <div class="tabs" style="margin-bottom:16px">
        ${runs.map(r => `
          <button class="tab ${r.run_id === best.run_id ? 'on' : ''}" data-rtab="${r.run_id}">
            ${r.config_name}
          </button>`).join('')}
      </div>` : '';

    out.innerHTML = `
      <p class="small muted" style="margin-bottom:14px">
        ${multi
          ? `Compared ${runs.length} configurations on ${runs[0].n_cases} questions. Ranked by MRR, then precision@k.`
          : `One configuration, ${runs[0].n_cases} questions.`}
      </p>
      <div class="kpi-grid">${cards}</div>
      <div class="card">
        <div class="card-head"><h2>Per-question results</h2></div>
        ${tabs}
        <div id="res-cases"><div class="empty">Loading…</div></div>
      </div>`;

    $$('[data-rtab]').forEach(t => t.onclick = () => {
      $$('[data-rtab]').forEach(x => x.classList.remove('on'));
      t.classList.add('on');
      loadResultCases(t.dataset.rtab);
    });

    loadResultCases(best.run_id);
  } catch (e) {
    out.innerHTML = `<div class="banner banner-err">${e.message}</div>`;
  }
}

async function loadResultCases(runId) {
  const box = $('#res-cases');
  box.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const cases = await api(`/runs/${runId}/cases`);
    const failing = cases.filter(c => !c.passed).length;
    box.innerHTML = `
      ${failing
        ? `<div class="banner banner-warn">${failing} question${failing === 1 ? '' : 's'} failed.</div>`
        : `<div class="banner banner-info">Every question passed.</div>`}
      <div class="cases">${cases.map(caseHtml).join('')}</div>`;
  } catch (e) {
    box.innerHTML = `<div class="banner banner-err">${e.message}</div>`;
  }
}

function caseHtml(c) {
  return `
    <div class="case ${c.passed ? '' : 'fail'}">
      <div class="case-q">${esc(c.question)}</div>
      <div class="case-a">${esc(c.answer)}</div>
      <div class="scores">
        <span class="score">rr <b>${fmt(c.reciprocal_rank, 2)}</b></span>
        <span class="score">faith <b>${c.faithfulness ?? '—'}</b>/5</span>
        <span class="score">rel <b>${c.relevance ?? '—'}</b>/5</span>
        <span class="score">comp <b>${c.completeness ?? '—'}</b>/5</span>
        <span class="score">${Math.round(c.latency_ms)}ms</span>
        ${c.failure_mode && c.failure_mode !== 'none'
          ? `<span class="score" style="color:var(--warn)">${c.failure_mode}</span>` : ''}
      </div>
      ${c.reasoning ? `<div class="reason">${esc(c.reasoning)}</div>` : ''}
    </div>`;
}

/* --------------------------------------------------------------- history */

async function loadHistory() {
  try {
    state.runs = await api('/runs?limit=50');
  } catch (e) {
    banner(e.message);
    return;
  }

  if (!state.runs.length) {
    $('#hist-kpis').innerHTML =
      '<div class="empty" style="grid-column:1/-1">No runs yet. Start an evaluation to see results here.</div>';
    return;
  }

  const latest = state.runs[0];
  $('#hist-kpis').innerHTML = [
    ['Hit rate', fmt(latest.hit_rate)],
    ['MRR', fmt(latest.mrr)],
    ['Faithfulness', fmt(latest.avg_faithfulness, 2)],
    ['Pass rate', fmt(latest.pass_rate)],
    ['p95 latency', `${Math.round(latest.p95_latency_ms)}ms`],
  ].map(([l, v]) => `<div class="kpi"><div class="kpi-label">${l}</div><div class="kpi-value">${v}</div></div>`).join('');

  drawChart();
  drawRunsTable();

  const opts = state.runs.map(r => `<option value="${r.run_id}">${r.run_id} · ${r.config_name}</option>`).join('');
  $('#cmp-a').innerHTML = opts;
  $('#cmp-b').innerHTML = opts;
  $('#det-run').innerHTML = opts;
  if (state.runs.length > 1) $('#cmp-a').value = state.runs[1].run_id;
  loadDetail();
}

function drawChart() {
  const ordered = [...state.runs].reverse();
  const mk = (label, key, color) => ({
    label, data: ordered.map(r => r[key]),
    borderColor: color, backgroundColor: color + '20',
    tension: .3, pointRadius: 3, pointHoverRadius: 6, borderWidth: 2,
  });

  if (state.chart) state.chart.destroy();
  state.chart = new Chart($('#trend-chart'), {
    type: 'line',
    data: {
      labels: ordered.map(r => `${r.config_name} · ${r.run_id.slice(0, 5)}`),
      datasets: [
        mk('Hit rate', 'hit_rate', '#4d8dff'),
        mk('MRR', 'mrr', '#3ecf8e'),
        mk('Precision@k', 'precision_at_k', '#f5a623'),
        mk('Pass rate', 'pass_rate', '#b47cf5'),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { beginAtZero: true, max: 1.05, grid: { color: '#222735' }, ticks: { color: '#98a1b3', font: { size: 11 } } },
        x: { grid: { display: false }, ticks: { color: '#98a1b3', font: { size: 10 }, maxRotation: 30 } },
      },
      plugins: {
        legend: { labels: { color: '#e6e9ef', usePointStyle: true, boxWidth: 8, font: { size: 12 } } },
        tooltip: { backgroundColor: '#181c25', borderColor: '#2f3648', borderWidth: 1, titleColor: '#e6e9ef', bodyColor: '#98a1b3', padding: 10 },
      },
    },
  });
}

function drawRunsTable() {
  $('#runs-table tbody').innerHTML = state.runs.map(r => `
    <tr>
      <td><span class="pill">${r.run_id}</span></td>
      <td>${r.config_name}</td>
      <td class="num">${fmt(r.hit_rate)}</td>
      <td class="num">${fmt(r.mrr)}</td>
      <td class="num">${fmt(r.precision_at_k)}</td>
      <td class="num">${fmt(r.avg_faithfulness, 2)}</td>
      <td class="num">${fmt(r.pass_rate)}</td>
      <td class="num">${Math.round(r.p50_latency_ms)}ms</td>
      <td class="num dim">${r.started_at.slice(0, 16).replace('T', ' ')}</td>
    </tr>`).join('');
}

async function doCompare() {
  const a = $('#cmp-a').value, b = $('#cmp-b').value;
  const out = $('#cmp-out');
  if (a === b) { out.innerHTML = '<div class="empty">Pick two different runs.</div>'; return; }

  try {
    const d = await api(`/compare/${a}/${b}`);
    const rows = [
      ['Hit rate', d.a_hit_rate, d.b_hit_rate, d.delta_hit_rate],
      ['MRR', d.a_mrr, d.b_mrr, d.delta_mrr],
      ['Faithfulness', d.a_faith, d.b_faith, d.delta_faith],
      ['Relevance', d.a_rel, d.b_rel, d.delta_rel],
      ['Pass rate', d.a_pass, d.b_pass, d.delta_pass],
    ];
    out.innerHTML = rows.map(([label, av, bv, dv]) => {
      const cls = dv > 0.0005 ? 'up' : dv < -0.0005 ? 'down' : 'flat';
      return `<div class="file-row">
        <span style="flex:1">${label}</span>
        <span class="mono">${fmt(av)} → ${fmt(bv)}</span>
        <span class="mono ${cls}" style="min-width:64px;text-align:right">${dv > 0 ? '+' : ''}${fmt(dv)}</span>
      </div>`;
    }).join('');
  } catch (e) {
    out.innerHTML = `<div class="banner banner-err">${e.message}</div>`;
  }
}

async function loadDetail() {
  const runId = $('#det-run').value;
  if (!runId) return;
  const out = $('#det-out');
  out.innerHTML = '<div class="empty">Loading…</div>';

  try {
    const cases = await api(`/runs/${runId}/cases`);
    const shown = $('#det-fails').checked ? cases.filter(c => !c.passed) : cases;
    out.innerHTML = shown.length
      ? shown.map(caseHtml).join('')
      : '<div class="empty">No cases to show.</div>';
  } catch (e) {
    out.innerHTML = `<div class="banner banner-err">${e.message}</div>`;
  }
}

/* ------------------------------------------------------------ listeners */

$('#drop').onclick = () => $('#file-input').click();
$('#file-input').onchange = e => { uploadFiles(e.target.files); e.target.value = ''; };

['dragenter', 'dragover'].forEach(ev =>
  $('#drop').addEventListener(ev, e => { e.preventDefault(); $('#drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(ev =>
  $('#drop').addEventListener(ev, e => { e.preventDefault(); $('#drop').classList.remove('over'); }));
$('#drop').addEventListener('drop', e => uploadFiles(e.dataTransfer.files));

$('#btn-sample').onclick = loadSamples;
$('#btn-add-q').onclick = addQuestion;
$('#btn-run').onclick = startRun;
$('#btn-cancel').onclick = () => { clearInterval(state.poll); location.hash = '#/new'; };
$('#btn-compare').onclick = doCompare;
$('#det-run').onchange = loadDetail;
$('#det-fails').onchange = loadDetail;

$$('.tab[data-htab]').forEach(t => t.onclick = () => {
  $$('.tab[data-htab]').forEach(x => x.classList.remove('on'));
  t.classList.add('on');
  ['trend', 'runs', 'compare', 'detail'].forEach(n =>
    $(`#htab-${n}`).classList.toggle('hidden', n !== t.dataset.htab));
});

route();