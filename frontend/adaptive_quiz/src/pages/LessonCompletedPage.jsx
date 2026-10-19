/**
 * pages/LessonCompletedPage.jsx
 * Shown after the student finishes a lesson.
 * The student picks a lesson — Grade → textbook part → chapter — then starts the
 * 3-level assessment for it. Grade isn't restricted: O/L covers both grades, so a
 * Grade 11 student can revise Grade 10 chapters too.
 */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAllLessons } from '../api/lessonApi';
import { getAssessmentQuestions, getAdaptiveAssessmentQuestions } from '../api/assessmentApi';
import { getStudentHistory } from '../api/responseApi';
import { useAssessment } from '../context/AssessmentContext';
import Logo from '../components/brand/Logo';
import './LessonCompletedPage.css';

const CHOICE_KEY = 'stemLessonId';   // last picked lesson, also used by ReportPage
const OTHER = 'other';               // lessons that aren't textbook chapters
const PART_LABEL = { 1: 'Part I', 2: 'Part II' };

const groupOf = (l) => (l.grade ? String(l.grade) : OTHER);

const LessonCompletedPage = () => {
  const navigate = useNavigate();
  const { student, setStudent, setLesson: setCtxLesson, startAssessment } = useAssessment();

  const [lessons,           setLessons]           = useState([]);
  const [grade,             setGrade]             = useState('');
  const [part,              setPart]              = useState(1);
  const [lessonId,          setLessonId]          = useState('');
  const [loading,           setLoading]           = useState(true);
  const [starting,          setStarting]          = useState(false);
  const [error,             setError]             = useState('');
  const [previous,          setPrevious]          = useState({ lessonId: null, weakAreas: [], score: null });
  const [history,           setHistory]           = useState([]);   // all past reports, newest first

  // Every report this student has — so past reports open even for lessons whose
  // questions are not live right now
  useEffect(() => {
    const sid = student?.studentId || JSON.parse(localStorage.getItem('stemStudent') || '{}')?.studentId;
    if (!sid) return;
    getStudentHistory(sid)
      .then(res => setHistory(res.data || []))
      .catch(() => { /* no history yet */ });
  }, [student?.studentId]);

  // Restore student from localStorage on page refresh
  useEffect(() => {
    if (!student) {
      const stored = localStorage.getItem('stemStudent');
      if (stored) setStudent(JSON.parse(stored));
      else { navigate('/login'); return; }
    }
  }, []);

  // Load every lesson once, then re-select the last one the student picked
  useEffect(() => {
    getAllLessons()
      .then(res => {
        const list = res.data || [];
        if (!list.length) { setError('No lessons found. Please run the seed script.'); return; }
        setLessons(list);
        const saved = list.find(l => l.lessonId === localStorage.getItem(CHOICE_KEY));
        const initial = saved || list.find(l => l.questionCount > 0) || list[0];
        setGrade(groupOf(initial));
        setPart(initial.part || 1);
        setLessonId(initial.lessonId);
      })
      .catch(() => setError('Could not connect to server. Is the backend running?'))
      .finally(() => setLoading(false));
  }, []);

  const lesson = lessons.find(l => l.lessonId === lessonId) || null;
  const groups = useMemo(() => [...new Set(lessons.map(groupOf))], [lessons]);
  const chapters = lessons.filter(l =>
    groupOf(l) === grade && (grade === OTHER || (l.part || 1) === part));

  // Returning student: look up their weak areas for the selected lesson (adaptive mode)
  useEffect(() => {
    const sid = student?.studentId || JSON.parse(localStorage.getItem('stemStudent') || '{}')?.studentId;
    if (!sid || !lesson || !lesson.questionCount) return undefined;
    let cancelled = false;
    getAdaptiveAssessmentQuestions(lesson.lessonId, sid)
      .then(adaptive => {
        if (cancelled) return;
        setPrevious({
          lessonId: lesson.lessonId,
          weakAreas: adaptive.isAdaptive ? (adaptive.previousWeakAreas || []) : [],
          score: adaptive.previousScore ?? null,
        });
      })
      .catch(() => { /* no previous session — normal mode */ });
    return () => { cancelled = true; };
  }, [student, lesson?.lessonId]);

  // Only trust the adaptive info if it belongs to the lesson on screen
  const isCurrent = previous.lessonId === lesson?.lessonId;
  const previousWeakAreas = isCurrent ? previous.weakAreas : [];
  const previousScore     = isCurrent ? previous.score : null;

  const pickLesson = (l) => {
    setLessonId(l.lessonId);
    localStorage.setItem(CHOICE_KEY, l.lessonId);
    setError('');
  };

  const pickGrade = (g) => {
    setGrade(g);
    const first = lessons.find(l => groupOf(l) === g && (g === OTHER || (l.part || 1) === part))
               || lessons.find(l => groupOf(l) === g);
    if (first) { setPart(first.part || 1); pickLesson(first); }
  };

  const pickPart = (p) => {
    setPart(p);
    const first = lessons.find(l => groupOf(l) === grade && (l.part || 1) === p);
    if (first) pickLesson(first);
  };

  // Latest report per lesson (history is newest first) + attempt counts
  const pastByLesson = useMemo(() => {
    const out = new Map();
    history.forEach(r => {
      const cur = out.get(r.lessonId);
      out.set(r.lessonId, cur ? { ...cur, attempts: cur.attempts + 1 } : { ...r, attempts: 1 });
    });
    return [...out.values()];
  }, [history]);
  const lastForLesson = pastByLesson.find(r => r.lessonId === lesson?.lessonId) || null;

  const openReport = (report) => {
    const l = lessons.find(x => x.lessonId === report.lessonId);
    if (!l) return;
    setCtxLesson(l);
    localStorage.setItem(CHOICE_KEY, l.lessonId);
    navigate('/report', { state: { sessionId: report.sessionId } });
  };

  const handleStartAssessment = async () => {
    if (!lesson) return;
    setStarting(true);
    try {
      const sid = student?.studentId;
      // Use adaptive endpoint when a previous session exists (it returns same shape)
      const res = sid && previousWeakAreas.length > 0
        ? await getAdaptiveAssessmentQuestions(lesson.lessonId, sid)
        : await getAssessmentQuestions(lesson.lessonId);

      const formattedQuestions = {
        level1: res.levels.level1.questions || [],
        level2: res.levels.level2.questions || [],
        level3: res.levels.level3.questions || [],
      };

      startAssessment(lesson, formattedQuestions, previousWeakAreas, previousScore);
      navigate('/assessment');
    } catch {
      setError('Failed to load questions. Please try again.');
    } finally {
      setStarting(false);
    }
  };

  const signOut = () => {
    localStorage.removeItem('stemToken');
    localStorage.removeItem('stemStudent');
    window.location.assign('/');   // back to the STEM Learn LK main dashboard
  };

  const firstName = (student?.name || '').trim().split(/\s+/)[0] || 'there';
  const initials = (student?.name || '?').trim().split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();

  const topbar = (
    <header className="lc-topbar">
      <Logo />
      {student && (
        <div className="lc-user">
          <span className="lc-avatar" aria-hidden="true">{initials}</span>
          <span className="lc-user-meta"><strong>{student.name}</strong><small>{student.studentId}</small></span>
          <button type="button" className="lc-signout" onClick={signOut}>Sign out</button>
        </div>
      )}
    </header>
  );

  if (loading) return (
    <div className="lc-page">
      {topbar}
      <div className="lc-loader"><div className="spin-ring"/><p>Loading lessons…</p></div>
    </div>
  );

  const ready = !!lesson && lesson.questionCount > 0;
  const lastReport = lastForLesson || pastByLesson[0];

  return (
    <div className="lc-page">
      {topbar}

      <main className="lc-layout">
        {/* ── Left column ── */}
        <section className="lc-main">
          <div className="lc-hero">
            <span className="lc-eyebrow">Lesson complete</span>
            <h1>Nice work, {firstName}. Ready to check what you learned?</h1>
            <p>Pick a chapter and take a short three-level quiz. You'll get hints when you're stuck and a study plan at the end.</p>
            <div className="lc-actions">
              {lastReport && (
                <button type="button" className="lc-btn lc-btn--secondary" onClick={() => openReport(lastReport)}>
                  <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 16V9M10 16V4M16 16v-5" /></svg>
                  View my last report &amp; study plan
                </button>
              )}
              <button type="button" className="lc-btn lc-btn--ghost" onClick={() => navigate('/practice')}>
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 3h7l3 3v11H5z" /><path d="M8 10h5M8 13h3" /></svg>
                Practice with my own notes
              </button>
            </div>
          </div>

          {/* Adaptive mode: previous weak areas */}
          {previousWeakAreas.length > 0 && (
            <div className="lc-focus">
              <div className="lc-focus-icon" aria-hidden="true">
                <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7" /><circle cx="10" cy="10" r="3.5" /><circle cx="10" cy="10" r="0.8" /></svg>
              </div>
              <div>
                <h3>Adaptive mode is on</h3>
                <p>Your last attempt scored {previousScore !== null ? `${Math.round(previousScore)}%` : '—'}. This quiz will focus on:</p>
                <div className="lc-weak-tags">
                  {previousWeakAreas.map(area => <span key={area} className="lc-weak-tag">{area}</span>)}
                </div>
              </div>
            </div>
          )}

          {/* Lesson picker: Grade → textbook part → chapter */}
          <div className="lc-panel">
            <div className="lc-panel-head">
              <h2>Choose a chapter</h2>
              <div className="lc-seg" role="group" aria-label="Grade">
                {groups.map(g => (
                  <button key={g} type="button" className={grade === g ? 'active' : ''} onClick={() => pickGrade(g)}>
                    {g === OTHER ? 'Other' : `Grade ${g}`}
                  </button>
                ))}
              </div>
            </div>

            {grade !== OTHER && (
              <div className="lc-tabs" role="group" aria-label="Textbook part">
                {[1, 2].map(p => (
                  <button key={p} type="button" className={part === p ? 'active' : ''} onClick={() => pickPart(p)}>
                    Textbook {PART_LABEL[p]}
                  </button>
                ))}
              </div>
            )}

            <div className="lc-chapters">
              {chapters.map(l => (
                <button
                  key={l.lessonId}
                  type="button"
                  className={`lc-chapter ${l.lessonId === lessonId ? 'selected' : ''} ${l.questionCount ? '' : 'soon'}`}
                  onClick={() => pickLesson(l)}
                >
                  <span className="lc-chapter-no">{l.chapterNo ?? '•'}</span>
                  <span className="lc-chapter-title">{l.title}</span>
                  {l.questionCount > 0
                    ? <span className="lc-chapter-count">{l.questionCount} questions</span>
                    : <span className="lc-chapter-soon">Coming soon</span>}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* ── Right column: selected lesson + start ── */}
        <aside className="lc-side">
          {lesson && (
            <div className="lc-summary">
              <span className="lc-subject">{lesson.subject}</span>
              <h2>{lesson.title}</h2>
              <p className="lc-summary-desc">{lesson.description}</p>
              <dl className="lc-facts">
                <div><dt>Grade</dt><dd>{lesson.gradeLevel}</dd></div>
                <div><dt>Time</dt><dd>~{lesson.estimatedDuration} min</dd></div>
                <div><dt>Questions</dt><dd>9 of {lesson.questionCount || 0}</dd></div>
              </dl>

              <h3 className="lc-steps-title">How the quiz works</h3>
              <ol className="lc-steps">
                <li className="lvl-1"><span>1</span><div><strong>Basic</strong><small>3 recall questions</small></div></li>
                <li className="lvl-2"><span>2</span><div><strong>Concept</strong><small>3 understanding questions</small></div></li>
                <li className="lvl-3"><span>3</span><div><strong>Advanced</strong><small>3 application questions</small></div></li>
              </ol>

              <p className="lc-note">
                <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="6" width="10" height="8" rx="2" /><path d="m13 9 4-2v6l-4-2" /></svg>
                Your camera and answers help the system offer hints and support while you work.
              </p>

              {error && <div className="lc-error">{error}</div>}

              <button id="btn-start-assessment" className="btn-start" onClick={handleStartAssessment} disabled={!ready || starting}>
                {starting
                  ? <><span className="spinner-sm"/>Preparing your quiz…</>
                  : ready ? <>Start assessment<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 10h10M11 6l4 4-4 4" /></svg></>
                          : 'Questions coming soon'}
              </button>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
};

export default LessonCompletedPage;
