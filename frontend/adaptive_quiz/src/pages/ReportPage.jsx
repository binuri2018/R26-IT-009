/**
 * pages/ReportPage.jsx
 * Final assessment report showing all aggregated metrics:
 * - Total score, level-wise scores, concept-wise performance
 * - Most common expression, weak areas, hint usage, progress summary
 * - Learning state distribution
 */
import { useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAssessment }       from '../context/AssessmentContext';
import { getReport, getProgressOverTime, getStudentHistory } from '../api/responseApi';
import { getStudyPlan, saveStudyPlan, resetStudyPlan } from '../api/studyPlanApi';
import { getAllLessons } from '../api/lessonApi';
import {
  formatLevelScores, formatConceptPerformance, formatExpressionFrequency,
  formatLearningStates, getPerformanceBadge, formatTotalTime,
  generateStudyPlan, formatProgressHistory, generateWeeklyTimetable, formatTimeRange,
  toLocalISODate, TIME_SLOT_NAMES, calculateBadges,
} from '../utils/reportUtils';
import { getExpressionEmoji }  from '../utils/quizLogic';
import { useVoice } from '../context/VoiceContext';
import ResearchCharts      from '../components/research/ResearchCharts';
import Logo from '../components/brand/Logo';
import './ReportPage.css';

const SESSION_CFG = {
  review:    { bg: 'rgba(239,68,68,0.10)',   border: 'rgba(239,68,68,0.28)',   text: '#dc2626', icon: '📚' },
  practice:  { bg: 'rgba(14, 124, 102,0.10)',  border: 'rgba(14, 124, 102,0.28)',  text: '#0B6553', icon: '🔄' },
  quiz:      { bg: 'rgba(45, 154, 114,0.10)',  border: 'rgba(45, 154, 114,0.28)',  text: '#13866F', icon: '📝' },
  flashcard: { bg: 'rgba(245,158,11,0.10)',  border: 'rgba(245,158,11,0.28)',  text: '#d97706', icon: '⚡' },
  advanced:  { bg: 'rgba(34,197,94,0.10)',   border: 'rgba(34,197,94,0.28)',   text: '#16a34a', icon: '🚀' },
};

const INTENSITY_LABEL = { light: 'Light Load', medium: 'Medium Intensity', intensive: 'Intensive Mode' };
const INTENSITY_COLOR = { light: '#22c55e',    medium: '#f59e0b',          intensive: '#ef4444' };
const DURATION_SCALE  = { quick: 0.6, standard: 1.0, deep: 1.5 };

const expressionLabel = (e) => (e === 'unknown' ? 'No face detected' : e);

const ReportPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { student, lesson, sessionId, resetAssessment, setStudent, setLesson } = useAssessment();
  const { registerCommands, speak } = useVoice();

  const handleRetry = useCallback(() => {
    resetAssessment();
    navigate('/lesson-complete');
  }, [resetAssessment, navigate]);

  // Register Voice Commands
  useEffect(() => {
    const unregister = registerCommands('report-page', {
      'retry': () => { speak("Restarting assessment."); handleRetry(); },
      'restart': () => { speak("Restarting assessment."); handleRetry(); },
      'logout': () => { speak("Logging out."); localStorage.clear(); navigate('/login'); },
      'login': () => { navigate('/login'); }
    });
    return unregister;
  }, [navigate, registerCommands, speak, handleRetry]);

  const [report,       setReport]       = useState(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState('');
  const [progress,     setProgress]     = useState([]);
  const [durationPref, setDurationPref] = useState('standard');
  const [disabledDays, setDisabledDays] = useState(() => new Set());

  // Saved (database-persisted) custom study plan, and the working copy while editing it.
  const [customPlan,   setCustomPlan]   = useState(null);
  const [editSchedule, setEditSchedule] = useState(null);
  const [savingPlan,   setSavingPlan]   = useState(false);
  const [planMsg,      setPlanMsg]      = useState('');

  useEffect(() => {
    // Recover student/lesson when this page is opened directly (e.g. right after
    // login) instead of arriving from a just-finished quiz, so a returning
    // student can pull up their existing report/study plan on its own.
    const stored = student || JSON.parse(localStorage.getItem('stemStudent') || 'null');
    if (!stored) { navigate('/login'); return; }
    if (!student) setStudent(stored);

    const fetchReport = async () => {
      try {
        let activeLesson = lesson;
        if (!activeLesson) {
          const lessonsRes = await getAllLessons();
          // The lesson the student last picked — not just the first one in the list
          const picked = localStorage.getItem('stemLessonId');
          activeLesson = lessonsRes.data?.find(l => l.lessonId === picked) || lessonsRes.data?.[0] || null;
          if (!activeLesson) { setError('No lessons found.'); setLoading(false); return; }
          setLesson(activeLesson);
        }

        // The context's sessionId is the quiz this tab last ran — possibly another lesson
        // (ids look like "<student>_<lesson>_<ts>"). Only use it for that lesson; otherwise
        // open this lesson's latest session. Never send no session: the backend would
        // merge every attempt into one report.
        const belongs = (s) => s?.startsWith(`${stored.studentId}_${activeLesson.lessonId}_`);
        const picked = location.state?.sessionId;      // a row clicked in "My past reports"
        let reportSession = belongs(picked) ? picked : belongs(sessionId) ? sessionId : null;
        if (!reportSession) {
          // Latest report for this lesson from the student's history (newest first) —
          // works even when the lesson has no live questions right now
          const hist = await getStudentHistory(stored.studentId);
          reportSession = (hist.data || []).find(r => r.lessonId === activeLesson.lessonId)?.sessionId || null;
          if (!reportSession) {
            setError('No completed assessment for this lesson yet.');
            return;
          }
        }

        const res = await getReport(stored.studentId, activeLesson.lessonId, reportSession);
        setReport(res.data);
        // Load progress timeline in parallel (non-blocking — failure is silent)
        try {
          const prog = await getProgressOverTime(stored.studentId, activeLesson.lessonId);
          setProgress(prog.data || []);
        } catch { /* no history yet */ }
        // Load any saved custom plan for this lesson (non-blocking — falls back to auto-generated)
        try {
          const saved = await getStudyPlan(activeLesson.lessonId);
          setCustomPlan(saved);
        } catch { /* fall back to auto-generated plan */ }
      } catch (err) {
        setError(err.response?.data?.message || 'Failed to generate report. Complete an assessment first.');
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
  }, []);

  // Recompute "today" against the current date rather than trusting a possibly-stale saved flag.
  const withFreshIsToday = (schedule) => {
    const todayIso = toLocalISODate(new Date());
    return schedule.map(d => ({ ...d, isToday: d.isoDate === todayIso }));
  };

  if (loading) return (
    <div className="report-page">
      <div className="report-loading"><div className="spin-ring"/><p>Generating your report…</p></div>
    </div>
  );

  if (error) return (
    <div className="report-page">
      <div className="report-error"><p>⚠️ {error}</p><button onClick={() => navigate('/lesson-complete')}>← Go Back</button></div>
    </div>
  );

  if (!report) return null;

  const levelScores     = formatLevelScores(report.levelScores);
  const concepts        = formatConceptPerformance(report.conceptPerformance);
  const expressions     = formatExpressionFrequency(report.expressionFrequency);
  const learningStates  = formatLearningStates(report.learningStateDistribution);
  const autoTimetable   = generateWeeklyTimetable(report);
  const durationScale   = DURATION_SCALE[durationPref];
  const progressFormatted = formatProgressHistory(progress);
  const badges           = calculateBadges(report, progressFormatted);

  const isEditing  = editSchedule !== null;
  const isCustom   = isEditing || Boolean(customPlan);
  const baseSchedule = isEditing
    ? editSchedule
    : (customPlan ? withFreshIsToday(customPlan.schedule) : autoTimetable.schedule);
  const weeklyMinutesLive = baseSchedule.reduce(
    (sum, d) => sum + (d.isRestDay ? 0 : d.sessions.reduce((s, x) => s + Number(x.duration || 0), 0)),
    0
  );
  const intensityLabel = customPlan?.intensity || autoTimetable.intensity;
  const weekRangeLabelDisplay = `${baseSchedule[0].dateLabel} – ${baseSchedule[6].dateLabel}`;

  // Non-edit mode: purely cosmetic dim toggle (legacy). Edit mode: real rest-day toggle
  // that changes the saved schedule and the live weekly-minutes total.
  const toggleDayRest = (isoDate) => {
    if (isEditing) {
      setEditSchedule(prev => prev.map(d => d.isoDate !== isoDate ? d : { ...d, isRestDay: !d.isRestDay }));
    } else {
      setDisabledDays(prev => {
        const next = new Set(prev);
        next.has(isoDate) ? next.delete(isoDate) : next.add(isoDate);
        return next;
      });
    }
  };

  const handleStartEdit = () => { setPlanMsg(''); setEditSchedule(baseSchedule.map(d => ({ ...d, sessions: d.sessions.map(s => ({ ...s })) }))); };
  const handleCancelEdit = () => { setEditSchedule(null); setPlanMsg(''); };

  const updateSession = (isoDate, sessionId, patch) => setEditSchedule(prev => prev.map(d =>
    d.isoDate !== isoDate ? d : { ...d, sessions: d.sessions.map(s => s.id !== sessionId ? s : { ...s, ...patch }) }
  ));

  const removeSession = (isoDate, sessionId) => setEditSchedule(prev => prev.map(d => {
    if (d.isoDate !== isoDate) return d;
    const sessions = d.sessions.filter(s => s.id !== sessionId);
    return { ...d, sessions, isRestDay: sessions.length === 0 };
  }));

  const addSession = (isoDate) => setEditSchedule(prev => prev.map(d => d.isoDate !== isoDate ? d : {
    ...d,
    isRestDay: false,
    sessions: [
      ...d.sessions,
      {
        id: `${isoDate}-custom-${Math.random().toString(36).slice(2, 8)}`,
        topic: 'New Topic', type: 'practice', time: 'After School', duration: 30, priority: 'medium',
      },
    ],
  }));

  const handleSavePlan = async () => {
    setSavingPlan(true);
    setPlanMsg('');
    try {
      const schedule = editSchedule.map(({ date, ...rest }) => rest); // eslint-disable-line no-unused-vars
      const weeklyMinutes = schedule.reduce(
        (sum, d) => sum + (d.isRestDay ? 0 : d.sessions.reduce((s, x) => s + Number(x.duration || 0), 0)),
        0
      );
      const weekRangeLabel = `${schedule[0].dateLabel} – ${schedule[6].dateLabel}`;
      const saved = await saveStudyPlan(lesson.lessonId, {
        schedule, weeklyMinutes, weekRangeLabel,
        intensity: customPlan?.intensity || autoTimetable.intensity,
      });
      setCustomPlan(saved);
      setEditSchedule(null);
      setPlanMsg('✓ Plan saved');
    } catch {
      setPlanMsg('Failed to save — try again.');
    } finally {
      setSavingPlan(false);
    }
  };

  const handleResetPlan = async () => {
    setSavingPlan(true);
    setPlanMsg('');
    try {
      await resetStudyPlan(lesson.lessonId);
      setCustomPlan(null);
      setEditSchedule(null);
      setPlanMsg('Reset to auto-generated plan');
    } catch {
      setPlanMsg('Failed to reset — try again.');
    } finally {
      setSavingPlan(false);
    }
  };

  const scoreBadge = getPerformanceBadge(report.totalScore);
  const scorePct   = Math.round(report.totalScore);
  const reportDate = report.createdAt ? new Date(report.createdAt).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : '';
  const initials   = (student?.name || '?').trim().split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();

  return (
    <div className="report-page">
      {/* ── Top bar ── */}
      <header className="rp-topbar">
        <Logo />
        <div className="rp-topbar-actions">
          <button type="button" className="rp-link" onClick={() => navigate('/lesson-complete')}>
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12 5l-5 5 5 5" /></svg>
            All chapters
          </button>
          {student && <span className="rp-avatar" title={student.name}>{initials}</span>}
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="rp-hero">
        <div className="rp-hero-inner">
          <div className="rp-hero-text">
            <span className="rp-eyebrow"><i />Assessment report{reportDate ? ` · ${reportDate}` : ''}</span>
            <h1>{lesson?.title || 'Your report'}</h1>
            <p className="rp-hero-meta">{lesson?.description}{student?.name ? ` · ${student.name}` : ''}</p>
            <p className="rp-hero-summary">{report.progressSummary}</p>

            {badges.length > 0 && (
              <div className="rp-badges">
                {badges.map((b) => (
                  <span key={b.label} className="rp-badge" title={b.desc}>
                    <i style={{ background: b.color }} />{b.label}
                  </span>
                ))}
              </div>
            )}

            <div className="rp-hero-actions">
              <button type="button" className="rp-btn rp-btn--primary" onClick={handleRetry}>
                Retake this quiz
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10a6 6 0 1 0 2-4.5M4 4v3h3" /></svg>
              </button>
              <button id="btn-download-pdf" type="button" className="rp-btn rp-btn--ghost" onClick={() => window.print()}>
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 8V3h8v5M6 14H4V8h12v6h-2M6 12h8v5H6z" /></svg>
                Download / Print PDF
              </button>
            </div>
          </div>

          <div className="rp-hero-score">
            <div className="rp-ring" style={{ '--pct': scorePct, '--ring': scoreBadge.color }}>
              <div className="rp-ring-inner">
                <strong>{scorePct}<small>%</small></strong>
                <span style={{ color: scoreBadge.color }}>{scoreBadge.label}</span>
              </div>
            </div>
            <dl className="rp-stats">
              <div>
                <dt>Time taken</dt>
                <dd>{formatTotalTime(report.totalTime)}</dd>
              </div>
              <div>
                <dt>Hints used</dt>
                <dd>{report.hintUsageCount}</dd>
              </div>
              <div>
                <dt>Mostly felt</dt>
                <dd className="rp-cap">{expressionLabel(report.mostCommonExpression)}</dd>
              </div>
            </dl>
          </div>
        </div>
      </section>

      <div className="report-container">

        {/* Level-wise scores */}
        <div className="report-section">
          <h3 className="section-title">Level-Wise Performance</h3>
          <div className="level-bars">
            {levelScores.map(({ level, label, score }) => {
              const b = getPerformanceBadge(score);
              return (
                <div key={level} className="level-bar-row">
                  <span className="level-bar-label">L{level} — {label}</span>
                  <div className="level-bar-track">
                    <div className="level-bar-fill" style={{ width: `${score}%`, background: b.color }} />
                  </div>
                  <span className="level-bar-score" style={{ color: b.color }}>{score}%</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Concept-wise performance */}
        <div className="report-section">
          <h3 className="section-title">Concept-Wise Performance</h3>
          <div className="concept-grid">
            {concepts.map(({ concept, score }) => {
              const b = getPerformanceBadge(score);
              return (
                <div key={concept} className="concept-card" style={{ borderColor: b.color + '44' }}>
                  <div className="concept-name">{concept}</div>
                  <div className="concept-score" style={{ color: b.color }}>{score}%</div>
                  <div className="concept-bar-track">
                    <div className="concept-bar-fill" style={{ width: `${score}%`, background: b.color }} />
                  </div>
                  <div className="concept-badge" style={{ background: b.color + '22', color: b.color }}>{b.label}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Weak areas */}
        {report.weakAreas?.length > 0 && (
          <div className="report-section">
            <h3 className="section-title">Weak areas</h3>
            <div className="weak-area-list">
              {report.weakAreas.map((area) => (
                <div key={area} className="weak-area-chip">{area}</div>
              ))}
            </div>
            <p className="weak-area-note">These concepts scored below 60%. Focus revision here.</p>
          </div>
        )}

        {/* Expression analysis */}
        <div className="report-section">
          <h3 className="section-title">Facial Expression Analysis</h3>
          <div className="expression-grid">
            {expressions.map(({ expression, count }) => (
              <div key={expression} className={`expression-chip ${expression === report.mostCommonExpression ? 'dominant' : ''}`}>
                <span className="exp-emoji">{getExpressionEmoji(expression)}</span>
                <span className="exp-name">{expressionLabel(expression)}</span>
                <span className="exp-count">{count}×</span>
              </div>
            ))}
          </div>
        </div>

        {/* Personalized Study Plan (Research Component) */}
        <div className="report-section" style={{ borderLeft: '4px solid #0E7C66' }}>
          <h3 className="section-title">✨ Personalized Study Plan</h3>
          <p className="section-subtitle">Tailored recommendations based on your performance &amp; behavior signals</p>
          <div className="study-plan-cards">
            {generateStudyPlan(report).map((item, idx) => {
              const typeConfig = {
                conceptual: { color: '#ef4444', bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.22)',  badge: 'Review Needed',    badgeBg: 'rgba(239,68,68,0.15)' },
                behavioral: { color: '#13866F', bg: 'rgba(167,139,250,0.08)', border: 'rgba(167,139,250,0.22)', badge: 'Behavior Insight', badgeBg: 'rgba(167,139,250,0.15)' },
                resource:   { color: '#22c55e', bg: 'rgba(34,197,94,0.08)',  border: 'rgba(34,197,94,0.22)',  badge: 'Next Action',      badgeBg: 'rgba(34,197,94,0.15)' },
              };
              const cfg = typeConfig[item.type] || typeConfig.resource;
              return (
                <div key={idx} className="study-card" style={{ background: cfg.bg, borderColor: cfg.border }}>
                  <div className="study-card-header">
                    <span className="study-card-icon">{item.icon}</span>
                    <span className="study-card-badge" style={{ background: cfg.badgeBg, color: cfg.color }}>{cfg.badge}</span>
                  </div>
                  <div className="study-card-title" style={{ color: cfg.color }}>{item.title}</div>
                  <div className="study-card-content">{item.content}</div>
                  {(item.metric || item.deadline) && (
                    <div className="study-card-target">
                      {item.metric && <span className="study-card-metric">🎯 {item.metric}</span>}
                      {item.deadline && <span className="study-card-deadline">📅 by {item.deadline}</span>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Weekly Study Timetable */}
        <div className="report-section timetable-section">
          <h3 className="section-title">🗓 Weekly Study Timetable</h3>
          <p className="section-subtitle">
            {isCustom ? 'Your customized plan' : 'Auto-generated from your results'} · {weekRangeLabelDisplay}
            {!isEditing && ' · tap a day to skip it'}
          </p>

          {/* Intensity badge + weekly minutes */}
          <div className="timetable-meta">
            <span
              className="intensity-badge"
              style={{ background: INTENSITY_COLOR[intensityLabel] + '1a', color: INTENSITY_COLOR[intensityLabel], borderColor: INTENSITY_COLOR[intensityLabel] + '44' }}
            >
              {INTENSITY_LABEL[intensityLabel]}
            </span>
            <span className="timetable-weekly-min">
              ~{isCustom ? Math.round(weeklyMinutesLive) : Math.round(autoTimetable.weeklyMinutes * durationScale)} min / week
            </span>
            {isCustom && !isEditing && <span className="plan-custom-badge">✏️ Customized</span>}
          </div>

          {/* Duration preference selector — only meaningful for the auto-generated plan */}
          {!isCustom && !isEditing && (
            <div className="timetable-controls">
              <span className="timetable-controls-label">Session length:</span>
              {[
                { key: 'quick',    label: 'Quick',    note: '×0.6' },
                { key: 'standard', label: 'Standard', note: '×1.0' },
                { key: 'deep',     label: 'Deep',     note: '×1.5' },
              ].map(opt => (
                <button
                  key={opt.key}
                  className={`duration-btn${durationPref === opt.key ? ' active' : ''}`}
                  onClick={() => setDurationPref(opt.key)}
                >
                  {opt.label} <span className="duration-btn-note">{opt.note}</span>
                </button>
              ))}
            </div>
          )}

          {/* Customize / Save / Reset controls */}
          <div className="timetable-plan-actions">
            {!isEditing ? (
              <>
                <button className="plan-action-btn" onClick={handleStartEdit}>✏️ Customize Plan</button>
                {isCustom && (
                  <button className="plan-action-btn plan-action-btn-ghost" onClick={handleResetPlan} disabled={savingPlan}>
                    ↺ Reset to Auto-generated
                  </button>
                )}
              </>
            ) : (
              <>
                <button className="plan-action-btn plan-action-btn-primary" onClick={handleSavePlan} disabled={savingPlan}>
                  {savingPlan ? 'Saving…' : '💾 Save Plan'}
                </button>
                <button className="plan-action-btn plan-action-btn-ghost" onClick={handleCancelEdit} disabled={savingPlan}>
                  Cancel
                </button>
              </>
            )}
            {planMsg && <span className="plan-action-msg">{planMsg}</span>}
          </div>

          {/* Day rows */}
          <div className="timetable-wrapper">
            {baseSchedule.map(({ day, short, dateLabel, isoDate, sessions, isRestDay, isToday }) => {
              const isDisabled = !isEditing && disabledDays.has(isoDate);
              return (
                <div
                  key={isoDate}
                  className={`timetable-day-row${isDisabled ? ' disabled' : ''}${isEditing ? ' editing' : ''}`}
                  onClick={() => toggleDayRest(isoDate)}
                  title={isEditing ? 'Click to toggle rest day' : (isDisabled ? `Click to re-enable ${day}` : `Click to skip ${day}`)}
                >
                  <div className={`timetable-day-label${isToday ? ' today' : ''}`}>
                    {short}
                    <span className="timetable-day-date">{dateLabel}</span>
                    {isToday && <span className="timetable-today-dot" />}
                  </div>

                  <div className="timetable-day-content">
                    {(isRestDay || sessions.length === 0) && (
                      <div className="timetable-rest">Rest &amp; Recharge</div>
                    )}

                    {!isRestDay && sessions.length > 0 && (
                      <div className="timetable-sessions">
                        {sessions.map(sess => {
                          const cfg  = SESSION_CFG[sess.type] || SESSION_CFG.practice;
                          const mins = isCustom ? Math.max(5, Math.round(sess.duration)) : Math.max(5, Math.round(sess.duration * durationScale));

                          if (isEditing) {
                            return (
                              <div key={sess.id} className="session-edit-pill" onClick={e => e.stopPropagation()}>
                                <div className="session-edit-row">
                                  <input
                                    className="session-edit-topic"
                                    type="text"
                                    value={sess.topic}
                                    onChange={e => updateSession(isoDate, sess.id, { topic: e.target.value })}
                                  />
                                  <button
                                    className="session-edit-remove"
                                    onClick={() => removeSession(isoDate, sess.id)}
                                    title="Remove session"
                                  >
                                    ×
                                  </button>
                                </div>
                                <div className="session-edit-row">
                                  <select value={sess.type} onChange={e => updateSession(isoDate, sess.id, { type: e.target.value })}>
                                    {Object.keys(SESSION_CFG).map(t => <option key={t} value={t}>{t}</option>)}
                                  </select>
                                  <select value={sess.time} onChange={e => updateSession(isoDate, sess.id, { time: e.target.value })}>
                                    {TIME_SLOT_NAMES.map(t => <option key={t} value={t}>{t}</option>)}
                                  </select>
                                  <input
                                    type="number" min={5} max={120} step={5}
                                    value={sess.duration}
                                    onChange={e => updateSession(isoDate, sess.id, { duration: Math.max(5, Number(e.target.value) || 5) })}
                                  />
                                  <span className="session-edit-min">min</span>
                                </div>
                              </div>
                            );
                          }

                          return (
                            <div
                              key={sess.id}
                              className="session-pill"
                              style={{ background: cfg.bg, borderColor: cfg.border, color: cfg.text }}
                              onClick={e => e.stopPropagation()}
                            >
                              <span className="session-pill-icon">{cfg.icon}</span>
                              <span className="session-pill-topic">{sess.topic}</span>
                              <span className="session-pill-sep">·</span>
                              <span className="session-pill-time">{formatTimeRange(sess.time, mins)}</span>
                              <span className="session-pill-sep">·</span>
                              <span className="session-pill-duration">{mins}m</span>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {isEditing && (
                      <button className="add-session-btn" onClick={e => { e.stopPropagation(); addSession(isoDate); }}>
                        + Add session
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <p className="timetable-tip">
            {isEditing
              ? 'Edit topic, type, time slot, and length for each session. Click a day to mark it a rest day. Save when done.'
              : 'Tap any day to mark it as a rest day. Your total weekly commitment updates automatically.'}
          </p>
        </div>

        {/* Progress Over Time */}
        {progress.length > 1 && (
          <div className="report-section" style={{ borderLeft: '4px solid #22c55e' }}>
            <h3 className="section-title">📈 Progress Over Time</h3>
            <p className="section-subtitle">{progress.length} sessions tracked — your learning journey</p>
            <div className="progress-timeline-v2">
              {progressFormatted.map((entry, idx) => {
                const b = getPerformanceBadge(entry.totalScore);
                const isLatest = idx === progress.length - 1;
                return (
                  <div key={entry.sessionIndex} className={`progress-card${isLatest ? ' progress-card-latest' : ''}`}>
                    <div className="progress-card-top">
                      <div className="progress-card-session">
                        <span className="progress-card-dot" style={{ background: b.color }} />
                        Session {entry.sessionIndex}
                        {isLatest && <span className="progress-card-latest-badge">Latest</span>}
                      </div>
                      {entry.date && <span className="progress-card-date">{entry.date}</span>}
                    </div>
                    <div className="progress-card-body">
                      <div className="progress-card-score-row">
                        <span className="progress-card-score" style={{ color: b.color }}>{entry.totalScore}%</span>
                        <span className="progress-card-badge-label" style={{ background: b.color + '22', color: b.color }}>{b.label}</span>
                        {entry.scoreDelta !== null && (
                          <span className={`progress-card-delta ${entry.scoreDelta >= 0 ? 'up' : 'down'}`}>
                            {entry.scoreDelta >= 0 ? '▲' : '▼'} {Math.abs(entry.scoreDelta)}%
                          </span>
                        )}
                      </div>
                      <div className="progress-card-bar-track">
                        <div className="progress-card-bar-fill" style={{ width: `${entry.totalScore}%`, background: b.color }} />
                      </div>
                      {(entry.improvedConcepts.length > 0 || entry.persistentWeakAreas.length > 0) && (
                        <div className="progress-card-tags">
                          {entry.improvedConcepts.map((c) => (
                            <span key={c} className="progress-tag-improved">✓ {c}</span>
                          ))}
                          {entry.persistentWeakAreas.map((c) => (
                            <span key={c} className="progress-tag-weak">⚠ {c}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Research Visualization */}
        <ResearchCharts responses={report.responses} />

        {/* Learning state distribution */}
        <div className="report-section">
          <h3 className="section-title">Learning State Distribution</h3>
          <div className="state-bars">
            {learningStates.map(({ state, label, count, color }) => {
              const total = Object.values(report.learningStateDistribution || {}).reduce((a, b) => a + b, 0);
              const pct   = total ? Math.round((count / total) * 100) : 0;
              return (
                <div key={state} className="state-bar-row">
                  <span className="state-bar-label" style={{ color }}>{label}</span>
                  <div className="state-bar-track">
                    <div className="state-bar-fill" style={{ width: `${pct}%`, background: color }} />
                  </div>
                  <span className="state-bar-pct" style={{ color }}>{count} ({pct}%)</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Actions */}
        <div className="report-actions">
          <button id="btn-retry" className="btn-retry" onClick={handleRetry}>🔄 Retry Assessment</button>
          <button id="btn-logout" className="btn-logout" onClick={() => { localStorage.clear(); navigate('/login'); }}>← Logout</button>
        </div>

      </div>
    </div>
  );
};

export default ReportPage;
