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

  if (loading) return (
    <div className="lc-page">
      <div className="lc-loader"><div className="spin-ring"/><p>Loading lessons…</p></div>
    </div>
  );

  const ready = !!lesson && lesson.questionCount > 0;

  return (
    <div className="lc-page">
      <div className="lc-card">
        {/* Congrats banner */}
        <div className="lc-banner">
          <div className="lc-trophy">🏆</div>
          <h1 className="lc-title">Lesson Complete!</h1>
          <p className="lc-subtitle">Great work! Now let's see how well you've understood the material.</p>
        </div>

        {/* Student info */}
        {student && (
          <div className="lc-student-chip">
            <span>👤</span>
            <span>{student.name} — {student.studentId}</span>
          </div>
        )}

        {/* Returning student: jump straight to existing report / study plan without redoing the quiz */}
        {(lastForLesson || pastByLesson[0]) && (
          <button
            type="button"
            className="lc-view-plan-link"
            onClick={() => openReport(lastForLesson || pastByLesson[0])}
            style={{
              display: 'block', margin: '0 auto 16px', background: 'none', border: 'none',
              color: '#6366f1', textDecoration: 'underline', cursor: 'pointer', fontSize: '14px',
            }}
          >
            📊 View My Last Report &amp; Study Plan
          </button>
        )}

        <button
          type="button"
          className="lc-view-plan-link"
          onClick={() => navigate('/practice')}
          style={{
            display: 'block', margin: '0 auto 16px', background: 'none', border: 'none',
            color: '#22c55e', textDecoration: 'underline', cursor: 'pointer', fontSize: '14px',
          }}
        >
          📝 Practice Mode — Generate a Quiz from Your Own Notes
        </button>

        {/* Adaptive mode: show previous weak areas if available */}
        {previousWeakAreas.length > 0 && (
          <div className="lc-adaptive-banner">
            <div className="lc-adaptive-title">
              🎯 Adaptive Mode — Focus on Weak Areas
            </div>
            <p className="lc-adaptive-note">
              Based on your last attempt ({previousScore !== null ? `${Math.round(previousScore)}%` : ''}), these concepts will be prioritised in this quiz:
            </p>
            <div className="lc-weak-tags">
              {previousWeakAreas.map((area) => (
                <span key={area} className="lc-weak-tag">📌 {area}</span>
              ))}
            </div>
          </div>
        )}

        {/* Lesson picker: Grade → textbook part → chapter */}
        <div className="lc-picker">
          <h3>📚 Choose your lesson</h3>

          <div className="lc-seg" role="group" aria-label="Grade">
            {groups.map(g => (
              <button key={g} type="button" className={grade === g ? 'active' : ''} onClick={() => pickGrade(g)}>
                {g === OTHER ? 'Other' : `Grade ${g}`}
              </button>
            ))}
          </div>

          {grade !== OTHER && (
            <div className="lc-seg" role="group" aria-label="Textbook part">
              {[1, 2].map(p => (
                <button key={p} type="button" className={part === p ? 'active' : ''} onClick={() => pickPart(p)}>
                  📖 {PART_LABEL[p]}
                </button>
              ))}
            </div>
          )}

          <div className="lc-chapters">
            {chapters.map(l => (
              <button
                key={l.lessonId}
                type="button"
                className={`lc-chapter ${l.lessonId === lessonId ? 'selected' : ''}`}
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

        {/* Lesson card */}
        {lesson && (
          <div className="lc-lesson-info">
            <div className="lc-lesson-badge">{lesson.subject}</div>
            <h2 className="lc-lesson-title">{lesson.title}</h2>
            <p className="lc-lesson-desc">{lesson.description}</p>
            <div className="lc-meta">
              <span>📚 {lesson.gradeLevel}</span>
              <span>⏱ ~{lesson.estimatedDuration} min</span>
            </div>
          </div>
        )}

        {/* Assessment info */}
        <div className="lc-assessment-info">
          <h3>What's Next?</h3>
          <div className="lc-levels">
            {[
              { lvl: 1, icon: '🟢', label: 'Level 1', desc: '3 Basic Questions' },
              { lvl: 2, icon: '🟡', label: 'Level 2', desc: '3 Concept Questions' },
              { lvl: 3, icon: '🔴', label: 'Level 3', desc: '3 Advanced Questions' },
            ].map(l => (
              <div className="lc-level-item" key={l.lvl}>
                <span>{l.icon}</span>
                <div>
                  <strong>{l.label}</strong>
                  <span>{l.desc}</span>
                </div>
              </div>
            ))}
          </div>
          <p className="lc-note">💡 The system will monitor your responses and facial expressions to provide personalised support.</p>
        </div>

        {error && <div className="lc-error">⚠️ {error}</div>}

        <button id="btn-start-assessment" className="btn-start" onClick={handleStartAssessment} disabled={!ready || starting}>
          {starting
            ? <><span className="spinner-sm"/>Preparing Assessment…</>
            : ready ? '🚀 Start Assessment' : '⏳ Questions for this lesson are coming soon'}
        </button>
      </div>
    </div>
  );
};

export default LessonCompletedPage;
