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

function renderDots() {
  const root = $("q-dots");
  root.innerHTML = surveyQuestions.map((_, i) =>
    `<span class="dot${i === qIndex ? " on" : ""}${answers[surveyQuestions[i].id] ? " done" : ""}"></span>`
  ).join("");
}

function renderQuestion() {
  const q = surveyQuestions[qIndex];
  if (!q) return;
  $("q-count").textContent = `Question ${qIndex + 1} of ${surveyQuestions.length}`;
  $("q-prompt").textContent = q.prompt;
  const picked = answers[q.id];
  $("q-options").innerHTML = (q.options || []).map((o) => {
    const sel = picked === o.value ? " selected" : "";
    return `<button type="button" class="choice-btn${sel}" data-value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</button>`;
  }).join("");
  $("q-options").querySelectorAll(".choice-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      answers[q.id] = btn.dataset.value;
      renderQuestion();
    });
  });
  $("btn-q-back").disabled = qIndex === 0;
  $("btn-q-next").textContent = qIndex === surveyQuestions.length - 1 ? "See my style" : "Next";
  renderDots();
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

function renderIntro(intro) {
  if (!intro || !Object.keys(intro).length) {
    $("tell-block").innerHTML = "";
    return;
  }
  let html = "";
  if (intro.concept_statement) {
    html += `<div class="card-note"><strong>The concept</strong>${escapeHtml(intro.concept_statement)}</div>`;
  }
  if (intro.explanation) {
    html += `<div class="card-note why"><strong>Why it works</strong>${escapeHtml(intro.explanation)}</div>`;
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
    renderIntro(data.science_intro || {});
    $("story-box").textContent = data.story || "";
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
