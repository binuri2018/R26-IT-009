const API = "";

const $ = (id) => document.getElementById(id);

let surveyQuestions = [];
let qIndex = 0;
let answers = {};
let profile = null;
let themeBlurbs = {};

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function showPanel(id) {
  ["step-survey", "step-theme", "step-topic"].forEach((pid) => {
    const el = $(pid);
    if (el) el.classList.toggle("hidden", pid !== id);
  });
  const phase = id === "step-survey" ? "survey" : id === "step-theme" ? "theme" : "topic";
  setPhase(phase);
}

function setPhase(phase) {
  const ws = $("workspace");
  if (!ws) return;
  ws.classList.remove("phase-survey", "phase-theme", "phase-topic", "phase-story");
  ws.classList.add(`phase-${phase}`);
}

async function refreshStatus() {
  const pill = $("health-pill");
  try {
    const r = await fetch(`${API}/api/narrative-learning/status`);
    const j = await r.json();
    pill.textContent = j.status === "ready" ? "Ready" : "Unavailable";
    pill.className = j.status === "ready" ? "health-pill ok" : "health-pill bad";
  } catch {
    pill.textContent = "API offline";
    pill.className = "health-pill bad";
  }
}

function renderProgress() {
  const pct = surveyQuestions.length ? ((qIndex + 1) / surveyQuestions.length) * 100 : 0;
  $("q-bar-fill").style.width = `${pct}%`;
}

function renderQuestion() {
  const q = surveyQuestions[qIndex];
  if (!q) return;
  $("q-count").textContent = `Question ${qIndex + 1} of ${surveyQuestions.length}`;
  $("q-prompt").textContent = q.prompt;
  const picked = answers[q.id];
  $("q-options").innerHTML = (q.options || []).map((o, i) => {
    const sel = picked === o.value ? " selected" : "";
    return `<button type="button" class="choice-btn${sel}" data-value="${escapeHtml(o.value)}">`
      + `<span class="choice-letter">${String.fromCharCode(65 + i)}</span>`
      + `<span class="choice-text">${escapeHtml(o.label)}</span></button>`;
  }).join("");
  $("q-options").querySelectorAll(".choice-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      answers[q.id] = btn.dataset.value;
      renderQuestion();
    });
  });
  $("btn-q-back").disabled = qIndex === 0;
  $("btn-q-next").textContent = qIndex === surveyQuestions.length - 1 ? "See my style" : "Next";
  renderProgress();
}

async function loadSurvey() {
  const r = await fetch(`${API}/api/narrative-learning/survey`);
  if (!r.ok) throw new Error("Could not load questions");
  const data = await r.json();
  surveyQuestions = data.questions || [];
  themeBlurbs = data.theme_blurbs || {};
  qIndex = 0;
  answers = {};
  renderQuestion();
}

async function loadChapters() {
  const r = await fetch(`${API}/api/narrative-learning/chapters`);
  if (!r.ok) throw new Error("Could not load chapters");
  const data = await r.json();
  const books = data.books || {};
  const bookSel = $("book");
  bookSel.innerHTML = "";
  Object.keys(books).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    bookSel.appendChild(opt);
  });
  bookSel.dataset.chapters = JSON.stringify(books);
  fillTopics();
}

function fillTopics() {
  const bookSel = $("book");
  const books = JSON.parse(bookSel.dataset.chapters || "{}");
  const topics = books[bookSel.value] || [];
  const topicSel = $("topic");
  topicSel.innerHTML = "";
  topics.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    topicSel.appendChild(opt);
  });
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTerms(text, definitions) {
  const html = escapeHtml(text);
  const byTerm = new Map();
  (definitions || []).forEach((d) => {
    const term = (d.term || "").trim();
    if (term.length > 1 && d.definition) byTerm.set(term.toLowerCase(), d);
  });
  if (!byTerm.size) return html;
  const terms = [...byTerm.keys()].sort((a, b) => b.length - a.length);
  const re = new RegExp(`\\b(${terms.map((t) => escapeRegex(escapeHtml(t))).join("|")})\\b`, "gi");
  return html.replace(re, (match) => {
    const d = byTerm.get(match.toLowerCase()) || byTerm.get(match.toLowerCase().replace(/&amp;/g, "&"));
    if (!d) return match;
    return `<button type="button" class="term" data-term="${escapeHtml(d.term)}" data-def="${escapeHtml(d.definition)}">${match}</button>`;
  });
}

function hideTermPop() {
  const pop = $("term-pop");
  if (pop) pop.remove();
}

function showTermPop(btn) {
  hideTermPop();
  const pop = document.createElement("div");
  pop.id = "term-pop";
  pop.className = "term-pop";
  pop.innerHTML = `<strong>${escapeHtml(btn.dataset.term)}</strong><span>${escapeHtml(btn.dataset.def)}</span>`;
  document.body.appendChild(pop);
  const r = btn.getBoundingClientRect();
  const w = pop.offsetWidth;
  const left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
  pop.style.left = `${left + window.scrollX}px`;
  pop.style.top = `${r.bottom + window.scrollY + 8}px`;
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest(".term[data-def]");
  if (btn) {
    const open = $("term-pop");
    if (open && open.dataset.for === btn.dataset.term) {
      hideTermPop();
    } else {
      showTermPop(btn);
      $("term-pop").dataset.for = btn.dataset.term;
    }
    return;
  }
  if (!e.target.closest("#term-pop")) hideTermPop();
});
window.addEventListener("resize", hideTermPop);

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function normalizeCheck(cq) {
  if (!cq || !cq.question || !Array.isArray(cq.options)) return null;
  const options = cq.options
    .map((o) => String(o || "").trim().replace(/^[A-Da-d][).:]\s+/, ""))
    .filter(Boolean);
  if (options.length < 2) return null;
  if ([cq.question, ...options].some((o) => /[<>]/.test(String(o)))) return null;
  let answer = cq.answer;
  if (typeof answer === "string" && !/^\d+$/.test(answer.trim())) {
    answer = options.findIndex((o) => o.toLowerCase() === answer.trim().toLowerCase());
  } else {
    answer = Number(answer);
  }
  if (!Number.isInteger(answer) || answer < 0 || answer >= options.length) return null;
  const correct = options[answer];
  const mixed = shuffle(options);
  return {
    question: String(cq.question),
    options: mixed,
    answer: mixed.indexOf(correct),
    explain: String(cq.explain || ""),
  };
}

function checkFromDefinitions(defs) {
  const usable = (defs || []).filter((d) => d.term && d.definition);
  if (usable.length < 2) return null;
  const [target, ...rest] = shuffle(usable);
  const options = shuffle([target, ...rest.slice(0, 2)]).map((d) => d.term);
  return {
    question: `Which word means: "${target.definition}"?`,
    options,
    answer: options.indexOf(target.term),
    explain: `${target.term}: ${target.definition}`,
  };
}

let storyCtx = null;

function newSessionId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function sendStoryFeedback(extra) {
  if (!storyCtx) return;
  fetch(`${API}/api/narrative-learning/story-feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...storyCtx, ...extra }),
  }).catch(() => {});
}

function resetRating() {
  $("rate-row").querySelectorAll(".rate-btn").forEach((b) => {
    b.classList.remove("selected");
    b.disabled = false;
  });
  $("rate-thanks").classList.add("hidden");
}

$("rate-row").querySelectorAll(".rate-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("rate-row").querySelectorAll(".rate-btn").forEach((b) => {
      b.classList.toggle("selected", b === btn);
      b.disabled = true;
    });
    $("rate-thanks").classList.remove("hidden");
    sendStoryFeedback({ event: "rating", rating: Number(btn.dataset.r) });
  });
});

function renderCheck(data) {
  const block = $("check-block");
  const fromModel = normalizeCheck(data.check_question);
  const cq = fromModel || checkFromDefinitions(data.key_definitions);
  const checkSource = fromModel ? "llm" : "definitions";
  const fb = $("check-feedback");
  fb.className = "check-feedback hidden";
  if (!cq) {
    block.classList.add("hidden");
    return;
  }
  $("check-question").textContent = cq.question;
  const opts = $("check-options");
  opts.innerHTML = cq.options.map((o, i) =>
    `<button type="button" class="choice-btn check-opt" data-i="${i}"><span class="choice-letter">${String.fromCharCode(65 + i)}</span><span class="choice-text">${escapeHtml(o)}</span></button>`
  ).join("");
  opts.querySelectorAll(".check-opt").forEach((btn) => {
    btn.addEventListener("click", () => {
      const picked = Number(btn.dataset.i);
      const right = picked === cq.answer;
      opts.querySelectorAll(".check-opt").forEach((b, i) => {
        b.disabled = true;
        if (i === cq.answer) b.classList.add("correct");
        else if (i === picked) b.classList.add("wrong");
      });
      fb.textContent = (right ? "Nice one! " : "Not quite. ") + cq.explain;
      fb.className = `check-feedback ${right ? "ok" : "bad"}`;
      sendStoryFeedback({ event: "check", check_correct: right, check_source: checkSource });
    });
  });
  block.classList.remove("hidden");
}

function renderIntro(intro, defs) {
  if (!intro || !Object.keys(intro).length) {
    $("tell-block").innerHTML = "";
    return;
  }
  let html = "";
  if (intro.concept_statement) {
    html += `<div class="card-note"><strong>The concept</strong>${highlightTerms(intro.concept_statement, defs)}</div>`;
  }
  if (intro.explanation) {
    html += `<div class="card-note why"><strong>Why it works</strong>${highlightTerms(intro.explanation, defs)}</div>`;
  }
  if (intro.equations && intro.equations.length) {
    html += `<div class="card-note eq"><strong>Key equations</strong>${intro.equations.map((e) => `<code>${escapeHtml(e)}</code>`).join(" &nbsp; ")}</div>`;
  }
  if (intro.real_world_note) {
    html += `<div class="card-note world"><strong>Real world</strong>${escapeHtml(intro.real_world_note)}</div>`;
  }
  $("tell-block").innerHTML = html;
}

function renderStudy(data) {
  const eqs = data.key_equations || [];
  const defs = data.key_definitions || [];
  const bullets = data.exam_bullets || [];
  let html = "";
  if (eqs.length) {
    html += `<p class="field-label">Equations</p>`;
    eqs.forEach((eq) => {
      html += `<div class="study-item"><div class="step-cap">${escapeHtml(eq.label || "")}</div><code>${escapeHtml(eq.equation || "")}</code></div>`;
    });
  }
  if (defs.length) {
    html += `<p class="field-label">Definitions</p>`;
    defs.forEach((d) => {
      html += `<div class="study-item"><strong>${escapeHtml(d.term || "")}</strong><div>${escapeHtml(d.definition || "")}</div></div>`;
    });
  }
  if (bullets.length) {
    html += `<p class="field-label">What to write in the exam</p>`;
    bullets.forEach((b, i) => {
      html += `<div class="study-item"><b>${i + 1}.</b> ${escapeHtml(b)}</div>`;
    });
  }
  $("study-block").innerHTML = html || `<p class="step-cap">Study notes will appear here.</p>`;
}

function renderSources(sources) {
  const el = $("sources-block");
  if (!sources || !sources.length) {
    el.innerHTML = `<p class="step-cap">No textbook passages attached.</p>`;
    return;
  }
  el.innerHTML = sources.map((s) => {
    const title = [s.filename, s.chapter].filter(Boolean).join(" · ");
    const meta = [`p. ${s.page || "?"}`, s.similarity || ""].filter(Boolean).join(" · ");
    return `<div class="source-card"><strong>${escapeHtml(title)}</strong><div class="step-cap">${escapeHtml(meta)}</div><div>${escapeHtml(s.snippet || "")}</div></div>`;
  }).join("");
}

async function goNext() {
  const q = surveyQuestions[qIndex];
  const err = $("error-box");
  err.classList.add("hidden");
  if (!answers[q.id]) {
    err.textContent = "Pick one, then tap Next.";
    err.classList.remove("hidden");
    return;
  }
  if (qIndex < surveyQuestions.length - 1) {
    qIndex += 1;
    renderQuestion();
    return;
  }
  await classifyStudent();
}

function goBack() {
  if (qIndex === 0) return;
  qIndex -= 1;
  renderQuestion();
}

function renderModelNote(data) {
  const box = $("theme-model");
  if (!data.predicted_theme) {
    box.classList.add("hidden");
    return;
  }
  const sure = data.model_confidence != null ? ` (${Math.round(data.model_confidence * 100)}% sure)` : "";
  const pred = `<strong>${escapeHtml(data.predicted_theme)}</strong>${sure}`;
  let text;
  if (data.method === "model-tiebreak") {
    text = `Your two answers pointed to different styles, so our classifier chose between them using your interests.`;
  } else if (data.method === "model") {
    text = `Our classifier picked this from your interests${sure}.`;
  } else if (data.model_agrees) {
    text = `Our classifier also predicted ${pred} from your interests and career goal. It agrees with you!`;
  } else {
    text = `From your interests, our classifier guessed ${pred}, but we went with the style you chose.`;
  }
  box.innerHTML = `<span class="theme-model-tag">ML classifier</span><span>${text}</span>`;
  box.classList.remove("hidden");
}

async function classifyStudent() {
  const err = $("error-box");
  err.classList.add("hidden");
  const btn = $("btn-q-next");
  btn.disabled = true;
  try {
    const r = await fetch(`${API}/api/narrative-learning/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(answers),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    profile = { ...answers, ...data };
    $("theme-name").textContent = data.theme;
    $("theme-blurb").textContent = data.blurb || themeBlurbs[data.theme] || "";
    renderModelNote(data);
    showPanel("step-theme");
  } catch (e) {
    err.textContent = e.message || "Could not pick a style.";
    err.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}

async function sendThemeFeedback(matched) {
  if (!profile) return;
  try {
    await fetch(`${API}/api/narrative-learning/theme-feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        theme: profile.theme,
        matched,
        interest: profile.interest || "",
        aspiration: profile.aspiration || "",
      }),
    });
  } catch {
    /* keep going even if log fails */
  }
  $("theme-summary").textContent = `Story style: ${profile.theme}`;
  showPanel("step-topic");
  $("study-panel").classList.remove("hidden");
  setPhase("topic");
}

async function generateStory() {
  const btn = $("btn-generate");
  const err = $("error-box-topic");
  err.classList.add("hidden");
  err.textContent = "";
  if (!profile) {
    err.textContent = "Please answer the questions first.";
    err.classList.remove("hidden");
    return;
  }
  btn.disabled = true;
  btn.textContent = "Writing…";
  $("results").classList.add("hidden");

  const body = {
    interest: profile.interest,
    aspiration: profile.aspiration,
    struggle_level: profile.struggle_level,
    grade: profile.grade || "",
    story_style: profile.story_style || "",
    story_opening: profile.story_opening || "",
    theme: profile.theme || "",
    book: $("book").value,
    topic: $("topic").value,
    diagnostic: $("diagnostic").value.trim() || "Explain the core concept in simple terms.",
  };

  try {
    const r = await fetch(`${API}/api/narrative-learning/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const detail = data.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : (detail || r.statusText);
      throw new Error(msg);
    }

    $("theme-pill").textContent = `Theme: ${data.theme || "—"}`;
    hideTermPop();
    storyCtx = {
      session_id: newSessionId(),
      model: data.model || "",
      theme: data.theme || "",
      book: body.book,
      topic: body.topic,
      grade: body.grade,
      struggle_level: body.struggle_level,
    };
    resetRating();
    renderIntro(data.science_intro || {}, data.key_definitions);
    $("story-box").innerHTML = highlightTerms(data.story || "", data.key_definitions);
    renderCheck(data);
    renderStudy(data);
    renderSources(data.sources || []);
    const quizTopic = encodeURIComponent(data.quiz_topic || body.topic);
    $("quiz-link").href = `/adaptive-quiz?topic=${quizTopic}`;
    $("step-topic").classList.add("hidden");
    $("results").classList.remove("hidden");
    $("study-panel").classList.remove("hidden");
    setPhase("story");
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    err.textContent = e.message || "Story generation failed.";
    err.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate personalized story";
  }
}

$("book").addEventListener("change", fillTopics);
$("btn-q-next").addEventListener("click", goNext);
$("btn-q-back").addEventListener("click", goBack);
$("btn-theme-yes").addEventListener("click", () => sendThemeFeedback(true));
$("btn-theme-no").addEventListener("click", () => sendThemeFeedback(false));
$("btn-generate").addEventListener("click", generateStory);
$("btn-back-survey").addEventListener("click", () => {
  $("study-panel").classList.add("hidden");
  $("results").classList.add("hidden");
  showPanel("step-survey");
  setPhase("survey");
  renderQuestion();
});

refreshStatus();
Promise.all([loadSurvey(), loadChapters()]).catch((e) => {
  $("error-box").textContent = e.message;
  $("error-box").classList.remove("hidden");
});
