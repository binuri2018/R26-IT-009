/**
 * components/teacher/QuestionReviewQueue.jsx
 * Human-in-the-loop review of PDF-generated questions. Generated questions are
 * saved as "pending" and students never see them until a teacher approves them
 * here — after fixing the wording, options, correct answer, hint or level if needed.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  getPendingQuestions, updateQuestion, approveQuestions, rejectQuestion,
} from '../../api/questionApi';

const LEVELS = {
  1: { accent: '#22c55e', bg: 'rgba(34,197,94,0.1)',  border: 'rgba(34,197,94,0.25)',  label: 'Level 1 — Basic' },
  2: { accent: '#f59e0b', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.25)', label: 'Level 2 — Concept' },
  3: { accent: '#0E7C66', bg: 'rgba(14, 124, 102,0.1)', border: 'rgba(14, 124, 102,0.25)', label: 'Level 3 — Advanced' },
};

const field = {
  width: '100%', boxSizing: 'border-box', background: '#fff',
  border: '1px solid #d1d5db', borderRadius: '8px', color: '#111827',
  padding: '8px 12px', fontSize: '13px', outline: 'none', fontFamily: 'inherit',
};

const small = { fontSize: '11px', fontWeight: '700', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.4px', margin: '12px 0 4px', display: 'block' };

const button = (bg, color, border) => ({
  background: bg, color, border: `1px solid ${border}`, borderRadius: '8px',
  padding: '7px 14px', fontSize: '12px', fontWeight: '700', cursor: 'pointer',
});

const errorMessage = (err, fallback) =>
  err.response?.data?.message || err.response?.data?.detail?.message || err.message || fallback;

const toDraft = (q) => ({
  questionText: q.questionText,
  options: [...q.options],
  correctIdx: Math.max(0, q.options.indexOf(q.correctAnswer)),
  hint: q.hint || '',
  shortTheoryExplanation: q.shortTheoryExplanation || '',
  quizLevel: q.quizLevel,
});

const isDirty = (q, d) =>
  d.questionText !== q.questionText ||
  d.options.join('\u0000') !== q.options.join('\u0000') ||
  d.options[d.correctIdx] !== q.correctAnswer ||
  d.hint !== (q.hint || '') ||
  d.shortTheoryExplanation !== (q.shortTheoryExplanation || '') ||
  Number(d.quizLevel) !== q.quizLevel;

export default function QuestionReviewQueue({ lessonId, refreshKey = 0, onCountChange }) {
  const [items,   setItems]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');
  const [notice,  setNotice]  = useState('');
  const [dirtyIds, setDirtyIds] = useState(() => new Set());
  const [busyAll, setBusyAll] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getPendingQuestions(lessonId)
      .then(res => { if (!cancelled) { setItems(res.data || []); setError(''); } })
      .catch(err => { if (!cancelled) setError(errorMessage(err, 'Could not load questions for review.')); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [lessonId, refreshKey]);
  useEffect(() => { onCountChange?.(items.length); }, [items.length, onCountChange]);

  const setDirty = useCallback((id, dirty) => {
    setDirtyIds(prev => {
      if (prev.has(id) === dirty) return prev;
      const next = new Set(prev);
      dirty ? next.add(id) : next.delete(id);
      return next;
    });
  }, []);

  const removeItem = (id, message) => {
    setItems(prev => prev.filter(q => q.questionId !== id));
    setDirty(id, false);
    if (message) setNotice(message);
  };

  const replaceItem = (updated) =>
    setItems(prev => prev.map(q => (q.questionId === updated.questionId ? updated : q)));

  const approveAll = async () => {
    if (!items.length || dirtyIds.size) return;
    if (!window.confirm(`Approve all ${items.length} question(s)? Students will start receiving them.`)) return;
    setBusyAll(true);
    setError('');
    try {
      const res = await approveQuestions(items.map(q => q.questionId));
      setItems([]);
      setNotice(res.message || 'All questions approved.');
    } catch (err) {
      setError(errorMessage(err, 'Approving failed.'));
    } finally {
      setBusyAll(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '12px', flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontWeight: '800', fontSize: '14px', color: '#111827' }}>
            Review before publishing {lessonId && <span style={{ color: '#0E7C66' }}>· {lessonId}</span>}
          </div>
          <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '2px' }}>
            AI-generated questions are hidden from students until you approve them. Fix anything that looks wrong first.
          </div>
        </div>
        {items.length > 0 && (
          <button
            onClick={approveAll}
            disabled={busyAll || dirtyIds.size > 0}
            title={dirtyIds.size ? 'Save or discard your edits first' : ''}
            style={{
              ...button('linear-gradient(135deg,#16a34a,#22c55e)', '#fff', '#16a34a'),
              opacity: busyAll || dirtyIds.size ? 0.5 : 1,
              cursor: busyAll || dirtyIds.size ? 'not-allowed' : 'pointer',
            }}
          >
            {busyAll ? 'Approving…' : `Approve all (${items.length})`}
          </button>
        )}
      </div>

      {notice && (
        <div style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.25)', color: '#16a34a', borderRadius: '8px', padding: '8px 12px', fontSize: '12px', marginBottom: '10px' }}>
          {notice}
        </div>
      )}
      {error && (
        <div style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', color: '#dc2626', borderRadius: '8px', padding: '8px 12px', fontSize: '12px', marginBottom: '10px' }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '24px', color: '#9ca3af', fontSize: '13px' }}>Loading questions…</div>
      ) : items.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '24px', color: '#9ca3af', fontSize: '13px' }}>
          Nothing waiting for review{lessonId ? ' in this lesson' : ''}.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {items.map((q, idx) => (
            <ReviewCard
              key={q.questionId}
              q={q}
              idx={idx}
              onDirty={setDirty}
              onSaved={replaceItem}
              onRemoved={removeItem}
              onError={setError}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ReviewCard({ q, idx, onDirty, onSaved, onRemoved, onError }) {
  const [draft, setDraft] = useState(() => toDraft(q));
  const [busy,  setBusy]  = useState('');
  const dirty = isDirty(q, draft);
  const lvl = LEVELS[Number(draft.quizLevel)] || LEVELS[1];

  useEffect(() => { onDirty(q.questionId, dirty); }, [dirty, q.questionId, onDirty]);

  const set = (key, value) => setDraft(d => ({ ...d, [key]: value }));
  const setOption = (i, value) => setDraft(d => {
    const options = [...d.options];
    options[i] = value;
    return { ...d, options };
  });

  const save = async () => {
    const updated = await updateQuestion(q.questionId, {
      questionText: draft.questionText,
      options: draft.options,
      correctAnswer: draft.options[draft.correctIdx],
      hint: draft.hint,
      shortTheoryExplanation: draft.shortTheoryExplanation,
      quizLevel: Number(draft.quizLevel),
    });
    onSaved(updated.data);
    setDraft(toDraft(updated.data));
  };

  const run = async (kind, action, fallback) => {
    setBusy(kind);
    onError('');
    try { await action(); }
    catch (err) { onError(errorMessage(err, fallback)); }
    finally { setBusy(''); }
  };

  const handleSave = () => run('save', save, 'Saving failed.');

  const handleApprove = () => run('approve', async () => {
    if (dirty) await save();          // publish exactly what the teacher sees
    await approveQuestions([q.questionId]);
    onRemoved(q.questionId, 'Question approved — students can now get it.');
  }, 'Approving failed.');

  const handleReject = () => {
    if (!window.confirm('Reject and delete this question?')) return;
    run('reject', async () => {
      await rejectQuestion(q.questionId);
      onRemoved(q.questionId, 'Question rejected.');
    }, 'Rejecting failed.');
  };

  return (
    <div style={{ background: 'rgba(15,23,42,0.03)', border: `1px solid ${dirty ? 'rgba(245,158,11,0.5)' : 'rgba(15,23,42,0.08)'}`, borderRadius: '12px', padding: '14px 16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', flexWrap: 'wrap' }}>
        <span style={{ background: lvl.bg, color: lvl.accent, border: `1px solid ${lvl.border}`, borderRadius: '8px', padding: '3px 8px', fontSize: '11px', fontWeight: '800' }}>
          Q{idx + 1}
        </span>
        <select
          value={draft.quizLevel}
          onChange={e => set('quizLevel', Number(e.target.value))}
          style={{ ...field, width: 'auto', padding: '4px 8px', fontSize: '12px', color: lvl.accent, fontWeight: '700' }}
        >
          {[1, 2, 3].map(l => <option key={l} value={l}>{LEVELS[l].label}</option>)}
        </select>
        <span style={{ fontSize: '11px', color: '#9ca3af' }}>{q.conceptTag} · ID {q.questionId}</span>
        {dirty && <span style={{ marginLeft: 'auto', fontSize: '11px', color: '#d97706', fontWeight: '700' }}>● unsaved edits</span>}
      </div>

      <label style={small}>Question</label>
      <textarea rows={2} value={draft.questionText} onChange={e => set('questionText', e.target.value)} style={{ ...field, resize: 'vertical' }} />

      <label style={small}>Options — select the correct answer</label>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {draft.options.map((opt, i) => {
          const correct = draft.correctIdx === i;
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input
                type="radio"
                name={`correct-${q.questionId}`}
                checked={correct}
                onChange={() => set('correctIdx', i)}
                title="Mark as correct answer"
              />
              <span style={{ fontSize: '12px', color: '#9ca3af', width: '14px' }}>{String.fromCharCode(65 + i)}</span>
              <input
                value={opt}
                onChange={e => setOption(i, e.target.value)}
                style={{
                  ...field,
                  borderColor: correct ? 'rgba(34,197,94,0.6)' : '#d1d5db',
                  background: correct ? 'rgba(34,197,94,0.06)' : '#fff',
                  fontWeight: correct ? '700' : '400',
                }}
              />
            </div>
          );
        })}
      </div>

      <label style={small}>Hint (shown when a student struggles)</label>
      <textarea rows={2} value={draft.hint} onChange={e => set('hint', e.target.value)} style={{ ...field, resize: 'vertical' }} />

      <label style={small}>Explanation (shown after a wrong answer)</label>
      <textarea rows={2} value={draft.shortTheoryExplanation} onChange={e => set('shortTheoryExplanation', e.target.value)} style={{ ...field, resize: 'vertical' }} />

      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', marginTop: '12px', flexWrap: 'wrap' }}>
        <button onClick={handleReject} disabled={!!busy} style={button('rgba(239,68,68,0.08)', '#dc2626', 'rgba(239,68,68,0.3)')}>
          {busy === 'reject' ? 'Rejecting…' : 'Reject'}
        </button>
        {dirty && (
          <>
            <button onClick={() => setDraft(toDraft(q))} disabled={!!busy} style={button('transparent', '#6b7280', '#d1d5db')}>
              ↺ Discard
            </button>
            <button onClick={handleSave} disabled={!!busy} style={button('rgba(14, 124, 102,0.1)', '#0B6553', 'rgba(14, 124, 102,0.3)')}>
              {busy === 'save' ? 'Saving…' : 'Save'}
            </button>
          </>
        )}
        <button onClick={handleApprove} disabled={!!busy} style={button('#16a34a', '#fff', '#16a34a')}>
          {busy === 'approve' ? 'Approving…' : dirty ? 'Save & approve' : 'Approve'}
        </button>
      </div>
    </div>
  );
}
