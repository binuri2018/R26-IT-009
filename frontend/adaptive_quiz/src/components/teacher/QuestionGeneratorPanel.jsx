import { useState, useRef, useEffect } from 'react';
import { generateQuestionsFromPDF, getPendingQuestions } from '../../api/questionApi';
import QuestionReviewQueue from './QuestionReviewQueue';

const API_URL = import.meta.env.VITE_API_URL || '/api/adaptive-quiz';

const card = {
  background: 'rgba(255,255,255,0.9)',
  backdropFilter: 'blur(12px)',
  border: '1px solid rgba(15,23,42,0.05)',
  borderRadius: '16px',
  padding: '28px',
};

const inputStyle = {
  width: '100%',
  background: 'rgba(15,23,42,0.05)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '10px',
  color: '#111827',
  padding: '10px 14px',
  fontSize: '14px',
  outline: 'none',
  boxSizing: 'border-box',
};

const labelStyle = {
  display: 'block',
  fontSize: '12px',
  fontWeight: '700',
  color: '#6b7280',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  marginBottom: '6px',
};

// "Grade 11 · Part I" → [chapters…], keeping the API's textbook order
function groupLessons(lessons) {
  return lessons.reduce((acc, l) => {
    const book = l.grade ? `Grade ${l.grade} · Part ${l.part === 2 ? 'II' : 'I'}` : 'Other lessons';
    (acc[book] = acc[book] || []).push(l);
    return acc;
  }, {});
}

const HISTORY_KEY = 'qgen_history';
const MAX_HISTORY = 10;

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

function saveToHistory(entry) {
  const prev = loadHistory();
  const next = [entry, ...prev.filter(h => h.id !== entry.id)].slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
}

export default function QuestionGeneratorPanel() {
  const [lessonId,      setLessonId]      = useState('');
  const [lessons,       setLessons]       = useState([]);
  const [conceptTag,    setConceptTag]    = useState('');
  const [numQuestions,  setNumQuestions]  = useState(9);
  const [pdfFile,       setPdfFile]       = useState(null);
  const [pdfName,       setPdfName]       = useState('');
  const [isDragging,    setIsDragging]    = useState(false);
  const [generating,    setGenerating]    = useState(false);
  const [result,        setResult]        = useState(null);
  const [error,         setError]         = useState('');
  const [showReview,    setShowReview]    = useState(false);
  const [reviewKey,     setReviewKey]     = useState(0);   // bump to reload the review queue
  const [pendingCount,  setPendingCount]  = useState(0);
  const [chapterInfo,   setChapterInfo]   = useState(null);  // which pages of the PDF were used
  /** Stem API JSON `cause`; `ml_unreachable` = TCP/service down; `ml_upstream_error` = model/HF/backend error inside ML */
  const [errorCause,    setErrorCause]    = useState(null);
  const [elapsed,       setElapsed]       = useState(0);
  const [history,       setHistory]       = useState(loadHistory);
  const [showHistory,   setShowHistory]   = useState(false);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);

  // Clear the elapsed-time interval if the panel unmounts mid-generation
  useEffect(() => {
    return () => clearInterval(timerRef.current);
  }, []);

  // Load lessons using plain fetch — avoids axiosInstance's 401 redirect interceptor
  // so the teacher dashboard works even without a student login token
  useEffect(() => {
    fetch(`${API_URL}/lessons`)
      .then(r => r.json())
      .then(data => {
        const list = data.data || [];
        setLessons(list);
        if (list.length > 0) setLessonId(list[0].lessonId);
      })
      .catch(() => {
        setError('Cannot reach backend on port 3001. Start it with: npm run dev');
      });
  }, []);

  // How many generated questions in this lesson still wait for approval
  useEffect(() => {
    const id = lessonId.trim();
    if (!id) return undefined;   // the Review button is hidden without a lesson
    const t = setTimeout(() => {
      getPendingQuestions(id)
        .then(res => setPendingCount(res.count || 0))
        .catch(() => setPendingCount(0));
    }, 400);
    return () => clearTimeout(t);
  }, [lessonId, reviewKey]);

  // ── File handling ────────────────────────────────────────────────────────

  const acceptFile = (file) => {
    if (!file) return;
    if (file.type !== 'application/pdf') { setError('Please select a PDF file.'); return; }
    if (file.size > 20 * 1024 * 1024)   { setError('PDF must be under 20 MB.');    return; }
    setPdfFile(file);
    setPdfName(file.name);
    setError(prev => (prev.includes('PDF') || prev.includes('20 MB')) ? '' : prev);
    setResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    acceptFile(e.dataTransfer.files[0]);
  };

  // ── Generate ─────────────────────────────────────────────────────────────

  const handleGenerate = async () => {
    if (!pdfFile)        { setError('Upload a PDF first.');         return; }
    if (!lessonId.trim()){ setError('Lesson ID is required.');      return; }

    setGenerating(true);
    setError('');
    setErrorCause(null);
    setResult(null);
    setElapsed(0);

    timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000);

    try {
      const data = await generateQuestionsFromPDF({
        pdf: pdfFile,
        lessonId: lessonId.trim(),
        conceptTag: conceptTag.trim() || 'General',
        numQuestions,
      });
      setResult(data.questions);
      setChapterInfo({ title: data.chapterTitle, pages: data.chapterPages });
      setReviewKey(k => k + 1);
      const entry = {
        id: Date.now(),
        lessonId: lessonId.trim(),
        conceptTag: conceptTag.trim() || 'General',
        pdfName,
        numQuestions,
        generatedAt: new Date().toISOString(),
        elapsed,
        questions: data.questions,
        total: (data.questions?.level1?.length || 0) + (data.questions?.level2?.length || 0) + (data.questions?.level3?.length || 0),
      };
      saveToHistory(entry);
      setHistory(loadHistory());
    } catch (err) {
      const data = err.response?.data;
      const msg =
        data?.message ||
        data?.detail ||
        err.message ||
        'Generation failed.';
      setError(msg);
      setErrorCause(data?.cause ?? null);
    } finally {
      clearInterval(timerRef.current);
      setGenerating(false);
    }
  };

  const reset = () => {
    setPdfFile(null);
    setPdfName('');
    setResult(null);
    setError('');
    setErrorCause(null);
    setElapsed(0);
  };

  const loadFromHistory = (entry) => {
    setLessonId(entry.lessonId);
    setConceptTag(entry.conceptTag);
    setNumQuestions(entry.numQuestions);
    setResult(entry.questions);
    setPdfName(entry.pdfName);
    setError('');
    setErrorCause(null);
    setShowHistory(false);
  };

  const deleteHistory = (id, e) => {
    e.stopPropagation();
    const next = loadHistory().filter(h => h.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
    setHistory(next);
  };

  // ── Derived ───────────────────────────────────────────────────────────────

  const totalSaved = result
    ? result.level1.length + result.level2.length + result.level3.length
    : 0;

  const legacyUnreachableHint =
    !errorCause &&
    /ECONNREFUSED|All connection attempts failed|Cannot reach the PDF service/i.test(error || '');
  const showMlStartInstructions = errorCause === 'ml_unreachable' || legacyUnreachableHint;
  const showHfClientHint =
    errorCause === 'ml_upstream_error' && /client has been closed/i.test(error || '');
  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div style={{ ...card, marginBottom: '32px' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h3 style={{ color: '#0E7C66', margin: 0, fontSize: '16px', fontWeight: '800' }}>
            Generate Questions from PDF
          </h3>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#6b7280' }}>
            Upload a lesson PDF — T5 model drafts MCQ questions for you to review before students see them
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {!result && lessonId.trim() && (
            <button onClick={() => setShowReview(v => !v)} style={{
              background: showReview ? 'rgba(245,158,11,0.15)' : 'rgba(15,23,42,0.05)',
              border: showReview ? '1px solid rgba(245,158,11,0.4)' : '1px solid rgba(255,255,255,0.1)',
              color: pendingCount ? '#d97706' : '#4b5563',
              borderRadius: '8px', padding: '6px 14px', fontSize: '12px',
              fontWeight: '700', cursor: 'pointer',
            }}>
              Review ({pendingCount})
            </button>
          )}
          {history.length > 0 && (
            <button onClick={() => setShowHistory(v => !v)} style={{
              background: showHistory ? 'rgba(14, 124, 102,0.15)' : 'rgba(15,23,42,0.05)',
              border: showHistory ? '1px solid rgba(14, 124, 102,0.4)' : '1px solid rgba(255,255,255,0.1)',
              color: showHistory ? '#0B6553' : '#4b5563',
              borderRadius: '8px', padding: '6px 14px', fontSize: '12px',
              fontWeight: '700', cursor: 'pointer',
            }}>
              History ({history.length})
            </button>
          )}
          {result && (
            <button onClick={reset} style={{
              background: 'rgba(15,23,42,0.05)', border: '1px solid rgba(255,255,255,0.1)',
              color: '#4b5563', borderRadius: '8px', padding: '6px 14px', fontSize: '12px',
              fontWeight: '700', cursor: 'pointer',
            }}>
              ↺ New
            </button>
          )}
        </div>
      </div>

      {/* History panel */}
      {showHistory && (
        <div style={{
          background: 'rgba(15,23,42,0.08)', border: '1px solid rgba(14, 124, 102,0.2)',
          borderRadius: '12px', padding: '16px', marginBottom: '20px',
        }}>
          <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '12px' }}>
            Previous Generations
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {history.map(h => (
              <div
                key={h.id}
                onClick={() => loadFromHistory(h)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '12px',
                  padding: '10px 14px', borderRadius: '10px', cursor: 'pointer',
                  background: 'rgba(15,23,42,0.05)', border: '1px solid rgba(15,23,42,0.05)',
                  transition: 'background 0.15s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(14, 124, 102,0.08)'}
                onMouseLeave={e => e.currentTarget.style.background = 'rgba(15,23,42,0.05)'}
              >
                <span style={{ fontSize: '20px' }}></span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '13px', fontWeight: '700', color: '#111827', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {h.pdfName || 'PDF file'}
                  </div>
                  <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>
                    <span style={{ color: '#0E7C66' }}>{h.lessonId}</span>
                    {' · '}
                    <span>{h.conceptTag}</span>
                    {' · '}
                    <span style={{ color: '#22c55e' }}>{h.total} questions</span>
                    {' · '}
                    <span>{new Date(h.generatedAt).toLocaleDateString()} {new Date(h.generatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                </div>
                <span style={{
                  fontSize: '11px', color: '#0E7C66', fontWeight: '700',
                  background: 'rgba(14, 124, 102,0.1)', border: '1px solid rgba(14, 124, 102,0.2)',
                  borderRadius: '6px', padding: '3px 8px', flexShrink: 0,
                }}>
                  Load
                </span>
                <button
                  onClick={(e) => deleteHistory(h.id, e)}
                  style={{
                    background: 'transparent', border: 'none', color: '#c2ccd9',
                    cursor: 'pointer', fontSize: '14px', padding: '2px 6px', borderRadius: '4px',
                    flexShrink: 0,
                  }}
                  title="Remove from history"
                >
                  
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {!result && showReview && (
        <div style={{
          border: '1px solid rgba(245,158,11,0.25)', borderRadius: '12px',
          padding: '16px', marginBottom: '20px', background: 'rgba(245,158,11,0.03)',
        }}>
          <QuestionReviewQueue lessonId={lessonId.trim()} refreshKey={reviewKey} onCountChange={setPendingCount} />
        </div>
      )}

      {!result ? (
        <>
          {/* Form row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: '16px', marginBottom: '20px' }}>
            <div>
              <label style={labelStyle}>Lesson *</label>
              {/* Pick from the textbook chapters students see — a typo'd free-text
                  lesson ID would send questions to a lesson no student can open */}
              <select
                style={inputStyle}
                value={lessonId}
                onChange={e => setLessonId(e.target.value)}
                disabled={generating || lessons.length === 0}
              >
                {lessons.length === 0 && <option value="">Loading lessons…</option>}
                {Object.entries(groupLessons(lessons)).map(([book, items]) => (
                  <optgroup key={book} label={book}>
                    {items.map(l => (
                      <option key={l.lessonId} value={l.lessonId}>
                        {l.chapterNo ? `${l.chapterNo}. ` : ''}{l.title}
                        {l.questionCount ? ` (${l.questionCount} live)` : ''}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Concept Tag</label>
              <input
                style={inputStyle}
                placeholder="e.g. Photosynthesis"
                value={conceptTag}
                onChange={e => setConceptTag(e.target.value)}
                disabled={generating}
              />
            </div>
            <div>
              <label style={labelStyle}>Questions</label>
              <select
                style={{ ...inputStyle, width: '90px' }}
                value={numQuestions}
                onChange={e => setNumQuestions(Number(e.target.value))}
                disabled={generating}
              >
                {[9, 18, 30, 50].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          </div>

          {/* Connection info — shows teacher exactly where questions will go */}
          {lessonId && (
            <div style={{
              background: 'rgba(14, 124, 102,0.08)', border: '1px solid rgba(14, 124, 102,0.2)',
              borderRadius: '10px', padding: '10px 14px', marginBottom: '16px',
              fontSize: '12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '15px' }}></span>
                <span style={{ color: '#4b5563' }}>
                  Questions saved to lesson <strong style={{ color: '#0B6553' }}>{lessonId}</strong>
                </span>
              </div>
              <div style={{ color: '#9ca3af', paddingLeft: '23px', lineHeight: '1.6' }}>
                Questions you approve go into a <strong style={{ color: '#6b7280' }}>question bank</strong>.
                Each student session randomly draws <strong style={{ color: '#6b7280' }}>3 per level (9 total)</strong> —
                so students get fresh questions every attempt.
              </div>
            </div>
          )}

          {/* Drop zone */}
          <div
            onClick={() => !generating && fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${isDragging ? '#0E7C66' : pdfFile ? '#22c55e' : 'rgba(15,23,42,0.05)'}`,
              borderRadius: '12px',
              padding: '32px 20px',
              textAlign: 'center',
              cursor: generating ? 'default' : 'pointer',
              background: isDragging ? 'rgba(14, 124, 102,0.06)' : pdfFile ? 'rgba(34,197,94,0.04)' : 'transparent',
              transition: 'all 0.2s ease',
              marginBottom: '20px',
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              style={{ display: 'none' }}
              onChange={e => acceptFile(e.target.files[0])}
            />
            {pdfFile ? (
              <div>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}></div>
                <div style={{ color: '#22c55e', fontWeight: '700', fontSize: '14px' }}>{pdfFile.name}</div>
                <div style={{ color: '#6b7280', fontSize: '12px', marginTop: '4px' }}>
                  {(pdfFile.size / 1024).toFixed(0)} KB — click to change
                </div>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: '36px', marginBottom: '10px', opacity: 0.4 }}></div>
                <div style={{ color: '#4b5563', fontSize: '14px', fontWeight: '600' }}>
                  Drop your PDF here or click to upload
                </div>
                <div style={{ color: '#9ca3af', fontSize: '12px', marginTop: '6px' }}>Max 20 MB</div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div style={{
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)',
              borderRadius: '10px', padding: '14px 16px', color: '#dc2626',
              fontSize: '13px', marginBottom: '16px', wordBreak: 'break-word',
            }}>
              <div style={{ fontWeight: '700', marginBottom: '6px' }}>{error}</div>
              {showHfClientHint && (
                <div style={{ color: '#d97706', fontSize: '12px', lineHeight: '1.7', marginBottom: '10px' }}>
                  The PDF service is running, but loading the T5 model from Hugging Face failed (often a one-time
                  download race or <code style={{ color: '#d97706' }}>--reload</code> interrupting the client).
                  Retry once; run without <code style={{ color: '#d97706' }}>--reload</code> until the model is cached
                  under your user cache.
                </div>
              )}
              {showMlStartInstructions && (
                <div style={{ color: '#dc2626', fontSize: '12px', lineHeight: '1.7' }}>
                  The PDF/ML API on port 8000 is not reachable. Start it from the repo (use your working Python/venv):
                  <pre style={{
                    background: 'rgba(15,23,42,0.08)', borderRadius: '6px',
                    padding: '8px 12px', margin: '6px 0 0', fontSize: '11px',
                    color: '#d97706', overflowX: 'auto',
                  }}>
{`cd ml-service
# example: same venv as STEM backend
..\\backend\\.venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000`}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={generating || !pdfFile}
            style={{
              width: '100%', padding: '14px',
              background: generating || !pdfFile
                ? 'rgba(14, 124, 102,0.2)'
                : 'linear-gradient(135deg, #0B6553, #13866F)',
              border: 'none', borderRadius: '12px',
              color: generating || !pdfFile ? '#0E7C66' : '#fff',
              fontSize: '15px', fontWeight: '800', cursor: generating || !pdfFile ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            {generating ? (
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                <span style={{
                  width: '16px', height: '16px',
                  border: '2px solid rgba(14, 124, 102,0.3)', borderTopColor: '#0E7C66',
                  borderRadius: '50%', animation: 'spin 0.8s linear infinite', display: 'inline-block',
                }} />
                {elapsed}s elapsed
              </span>
            ) : 'Generate Questions'}
          </button>

          {generating && <GeneratingSteps elapsed={elapsed} numQuestions={numQuestions} />}
        </>
      ) : (
        /* ── Results ── */
        <>
          {/* Generated → waiting for review banner */}
          <div style={{
            background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)',
            borderRadius: '12px', padding: '14px 18px',
            display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px',
          }}>
            <span style={{ fontSize: '24px' }}></span>
            <div>
              <div style={{ color: '#b45309', fontWeight: '800', fontSize: '14px' }}>
                {totalSaved} questions generated — waiting for your review
              </div>
              <div style={{ color: '#6b7280', fontSize: '12px', marginTop: '2px' }}>
                Lesson: <strong style={{ color: '#4b5563' }}>{lessonId}</strong>
                {' · '}Tag: <strong style={{ color: '#4b5563' }}>{conceptTag || 'General'}</strong>
                {' · '}Took {elapsed}s
                {chapterInfo?.title && (
                  <div style={{ marginTop: '4px' }}>
                    {chapterInfo.pages
                      ? <>Used only PDF pages <strong style={{ color: '#4b5563' }}>{chapterInfo.pages[0]}–{chapterInfo.pages[1]}</strong> (chapter “{chapterInfo.title}”)</>
                      : <>Chapter “{chapterInfo.title}” wasn't found as a section of this PDF — used the whole PDF.</>}
                  </div>
                )}
                {' · '}Students won't see them until you approve.
              </div>
            </div>
          </div>

          <QuestionReviewQueue lessonId={lessonId.trim()} refreshKey={reviewKey} onCountChange={setPendingCount} />
        </>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

const GEN_STEPS = [
  { from: 0,  to: 3,   label: 'Uploading PDF to ML service...',           icon: '📤' },
  { from: 3,  to: 8,   label: 'Extracting text from document...',          icon: '📄' },
  { from: 8,  to: 14,  label: 'Identifying answer candidates...',           icon: '🔍' },
  { from: 14, to: 999, label: 'T5 batch inference — generating MCQs...',   icon: '🤖' },
];

function GeneratingSteps({ elapsed, numQuestions }) {
  // Mini-batches of 16 questions, ~5s each + ~8s overhead (PDF + candidates)
  const estimate = Math.round(Math.ceil(numQuestions / 16) * 5 + 8);
  const currentStep = GEN_STEPS.findIndex(s => elapsed >= s.from && elapsed < s.to);
  const pct = Math.min(100, Math.round((elapsed / estimate) * 100));

  return (
    <div style={{ marginTop: '14px' }}>
      {/* Progress bar */}
      <div style={{ height: '4px', background: 'rgba(15,23,42,0.05)', borderRadius: '99px', marginBottom: '14px', overflow: 'hidden' }}>
        <div style={{
          height: '100%', borderRadius: '99px',
          width: `${pct}%`,
          background: 'linear-gradient(90deg, #0B6553, #13866F)',
          transition: 'width 1s linear',
        }} />
      </div>
      {/* Steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {GEN_STEPS.map((step, idx) => {
          const done    = elapsed >= step.to && step.to !== 999;
          const active  = idx === currentStep;
          const pending = !done && !active;
          return (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '14px', width: '20px', textAlign: 'center', opacity: pending ? 0.3 : 1 }}>
                {done ? '✅' : active ? step.icon : '○'}
              </span>
              <span style={{
                fontSize: '12px',
                color: done ? '#22c55e' : active ? '#111827' : '#c2ccd9',
                fontWeight: active ? '700' : '400',
              }}>
                {step.label}
              </span>
              {active && (
                <span style={{ fontSize: '11px', color: '#9ca3af', marginLeft: 'auto' }}>
                  ~{Math.max(0, estimate - elapsed)}s left
                </span>
              )}
            </div>
          );
        })}
      </div>
      <p style={{ textAlign: 'center', color: '#c2ccd9', fontSize: '11px', marginTop: '10px' }}>
        Estimated {estimate}s for {numQuestions} questions · batch inference active
      </p>
    </div>
  );
}
