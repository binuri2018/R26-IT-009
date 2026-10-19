import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { loginStudent, registerStudent } from '../api/authApi';
import { useAssessment } from '../context/AssessmentContext';
import { useVoice } from '../context/VoiceContext';
import Logo from '../components/brand/Logo';
import LottiePlayer from '../components/brand/LottiePlayer';
import adaptiveQuizAnimation from '../assets/lottie/adaptive-quiz.json';
import './LoginPage.css';

// Lowercase ("simple") letters only — no capitals anywhere in the address
const EMAIL_RE = /^[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$/;

// Per-field rules. Each returns an error string, or '' when the value is valid.
const FIELD_RULES = {
  studentId: (v) => {
    const s = v.trim();
    if (!s) return 'Student ID is required';
    if (s.length < 3) return 'Student ID must be at least 3 characters';
    if (/\s/.test(s)) return 'Student ID cannot contain spaces';
    return '';
  },
  name: (v) => {
    const s = v.trim();
    if (!s) return 'Full name is required';
    if (s.length < 2) return 'Name must be at least 2 characters';
    // Letters and spaces only (Sinhala/Tamil letters allowed) — no numbers or symbols
    if (!/^[\p{L}\p{M}]+(\s+[\p{L}\p{M}]+)*$/u.test(s)) return 'Name can contain letters and spaces only';
    return '';
  },
  email: (v) => {
    const s = v.trim();
    if (!s) return 'Email address is required';
    if (/[A-Z]/.test(s)) return 'Email must use simple (lowercase) letters only';
    if (!EMAIL_RE.test(s)) return 'Enter a valid email address';
    return '';
  },
  password: (v) => {
    if (!v) return 'Password is required';
    if (v.length < 6) return 'Password must be at least 6 characters';
    return '';
  },
};

const LoginPage = () => {
  const navigate = useNavigate();
  const { setStudent } = useAssessment();
  const { registerCommands, speak } = useVoice();

  const [mode, setMode]         = useState('login'); // 'login' | 'register'
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Form fields
  const [form, setForm] = useState({
    studentId: '', name: '', email: '', password: '', gradeLevel: 'Year 1',
  });
  // Per-field error messages, shown under each input.
  const [fieldErrors, setFieldErrors] = useState({});
  // Only surface a field's error once it's been touched (blur) or after a submit attempt.
  const [touched, setTouched] = useState({});

  // Register Voice Commands
  useEffect(() => {
    const unregister = registerCommands('login-page', {
      'register': () => { setMode('register'); speak("Switched to registration mode."); },
      'login': () => { setMode('login'); speak("Switched to login mode."); },
      'teacher': () => { speak("Navigating to teacher dashboard."); navigate('/teacher'); },
      'dashboard': () => { speak("Navigating to teacher dashboard."); navigate('/teacher'); }
    });
    return unregister;
  }, [navigate, registerCommands, speak]);

  // Which fields apply in the current mode.
  const activeFields = mode === 'register'
    ? ['studentId', 'name', 'email', 'password']
    : ['email', 'password'];

  const validateField = (field, value) => FIELD_RULES[field](value);

  const validateAll = () => {
    const next = {};
    for (const field of activeFields) {
      const msg = validateField(field, form[field]);
      if (msg) next[field] = msg;
    }
    return next;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
    setError('');
    // Live-clear a field's error as soon as it becomes valid again.
    if (touched[name]) {
      setFieldErrors((fe) => ({ ...fe, [name]: validateField(name, value) }));
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    setTouched((t) => ({ ...t, [name]: true }));
    setFieldErrors((fe) => ({ ...fe, [name]: validateField(name, value) }));
  };

  const switchMode = (next) => {
    setMode(next);
    setError('');
    setFieldErrors({});
    setTouched({});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const errs = validateAll();
    setFieldErrors(errs);
    setTouched(Object.fromEntries(activeFields.map((f) => [f, true])));
    if (Object.keys(errs).length > 0) {
      setError('Please fix the highlighted fields.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      let data;
      if (mode === 'login') {
        data = await loginStudent({ email: form.email.trim(), password: form.password });
      } else {
        data = await registerStudent({
          studentId: form.studentId.trim(),
          name: form.name.trim(),
          email: form.email.trim(),
          password: form.password,
          gradeLevel: form.gradeLevel,
        });
      }

      // Persist token and student info
      localStorage.setItem('stemToken',   data.data.token);
      localStorage.setItem('stemStudent', JSON.stringify(data.data));
      setStudent(data.data);
      navigate('/lesson-complete');
    } catch (err) {
      setError(err.response?.data?.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const showError = (field) => Boolean(fieldErrors[field] && touched[field]);
  const fieldClass = (field) => `form-group${showError(field) ? ' has-error' : ''}`;
  const renderFieldError = (field) =>
    showError(field) ? <span className="field-error">{fieldErrors[field]}</span> : null;

  return (
    <div className="auth">
      {/* ── Left: curved brand panel with Lottie ── */}
      <aside className="auth__side">
        <span className="auth__orb auth__orb--1" aria-hidden="true" />
        <span className="auth__orb auth__orb--2" aria-hidden="true" />
        <div className="auth__side-inner">
          <div className="auth__brand">
            <span className="auth__brand-tile"><Logo withText={false} size={40} /></span>
            <span className="auth__brand-name">STEM Learn LK</span>
          </div>

          <div className="auth__pitch">
            <span className="auth__eyebrow"><i />Adaptive assessment · Grade 10 &amp; 11 science</span>
            <h2>Quizzes that <em>adapt to every learner</em></h2>
            <p>
              Three-level quizzes from the national syllabus that read how each student is
              doing — and step in with hints and support the moment they get stuck.
            </p>
          </div>

          <LottiePlayer className="auth__lottie" animationData={adaptiveQuizAnimation} />

          <ul className="auth__bullets">
            <li><i className="teal" />Learning-state prediction</li>
            <li><i className="amber" />Emotion-aware hints</li>
          </ul>
        </div>
      </aside>

      {/* ── Right: sign-in ── */}
      <main className="auth__main">
        <div className="auth__panel">
          <div className="auth__badge"><Logo withText={false} size={54} /></div>
          <header className="auth__head">
            <h1>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
            <p>{mode === 'login'
              ? 'Sign in with your student account to continue'
              : 'Register to start your first adaptive quiz'}</p>
          </header>

          <form onSubmit={handleSubmit} className="auth__form" noValidate>
            {mode === 'register' && (
              <div className="auth__row">
                <div className={fieldClass('studentId')}>
                  <label htmlFor="studentId">Student ID</label>
                  <div className="field">
                    <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="4" width="14" height="12" rx="2" /><path d="M6 8h4M6 11h8" /></svg>
                    <input id="studentId" name="studentId" type="text" placeholder="e.g. ST22001234"
                           value={form.studentId} onChange={handleChange} onBlur={handleBlur}
                           aria-invalid={showError('studentId')} />
                  </div>
                  {renderFieldError('studentId')}
                </div>
                <div className={fieldClass('name')}>
                  <label htmlFor="name">Full name</label>
                  <div className="field">
                    <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="7" r="3.2" /><path d="M4 17c1-3.2 3.3-4.6 6-4.6s5 1.4 6 4.6" /></svg>
                    <input id="name" name="name" type="text" placeholder="Your full name"
                           value={form.name} onChange={handleChange} onBlur={handleBlur}
                           aria-invalid={showError('name')} />
                  </div>
                  {renderFieldError('name')}
                </div>
              </div>
            )}

            <div className={fieldClass('email')}>
              <label htmlFor="email">Email address</label>
              <div className="field">
                <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="5" width="14" height="10" rx="2" /><path d="m3.5 6 6.5 5 6.5-5" /></svg>
                <input id="email" name="email" type="email" placeholder="e.g. student@school.lk" autoComplete="email"
                       value={form.email} onChange={handleChange} onBlur={handleBlur}
                       aria-invalid={showError('email')} />
              </div>
              {renderFieldError('email')}
            </div>

            <div className={fieldClass('password')}>
              <label htmlFor="password">Password</label>
              <div className="field">
                <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="4" y="9" width="12" height="8" rx="2" /><path d="M7 9V7a3 3 0 0 1 6 0v2" /></svg>
                <input id="password" name="password" type={showPassword ? 'text' : 'password'} placeholder="••••••••"
                       autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                       value={form.password} onChange={handleChange} onBlur={handleBlur}
                       aria-invalid={showError('password')} />
                <button type="button" className="field__eye" onClick={() => setShowPassword(v => !v)}
                        aria-label={showPassword ? 'Hide password' : 'Show password'}>
                  <svg viewBox="0 0 20 20" aria-hidden="true">
                    <path d="M2 10s3-5.5 8-5.5S18 10 18 10s-3 5.5-8 5.5S2 10 2 10z" /><circle cx="10" cy="10" r="2.4" />
                    {showPassword && <path d="M3 3l14 14" />}
                  </svg>
                </button>
              </div>
              {renderFieldError('password')}
              {mode === 'register' && !showError('password') && (
                <span className="field-hint">At least 6 characters</span>
              )}
            </div>

            {error && <div className="form-error" role="alert">{error}</div>}

            <button id="btn-submit" type="submit" className="auth__submit" disabled={loading}>
              {loading ? <span className="spinner" /> : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>

          <p className="auth__switch">
            {mode === 'login' ? (
              <>New here? <button id="tab-register" type="button" onClick={() => switchMode('register')}>Create an account</button></>
            ) : (
              <>Already registered? <button id="tab-login" type="button" onClick={() => switchMode('login')}>Sign in</button></>
            )}
          </p>

          <p className="auth__teacher">
            Are you a teacher? <button type="button" onClick={() => navigate('/teacher')}>Open the Teacher Portal</button>
          </p>
        </div>
      </main>
    </div>
  );
};

export default LoginPage;
