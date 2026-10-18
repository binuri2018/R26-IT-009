/**
 * components/brand/Logo.jsx
 * STEM Learn LK mark: three rising level bars with an adaptive path climbing them.
 */
const Logo = ({ size = 36, withText = true, tone = 'dark' }) => (
  <span className={`brand brand--${tone}`}>
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <rect width="40" height="40" rx="11" fill="var(--brand-mark-bg, #0E7C66)" />
      <rect x="9"  y="23" width="5.5" height="8"  rx="1.6" fill="#ffffff" opacity="0.55" />
      <rect x="17.25" y="17" width="5.5" height="14" rx="1.6" fill="#ffffff" opacity="0.8" />
      <rect x="25.5" y="11" width="5.5" height="20" rx="1.6" fill="#ffffff" />
      <path d="M8 19.5 C 14 18, 18 12, 24 10.5 S 31 7.5, 33 7" fill="none"
            stroke="#F5B54A" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="33" cy="7" r="2.4" fill="#F5B54A" />
    </svg>
    {withText && (
      <span className="brand__text">
        STEM Learn <em>LK</em>
      </span>
    )}
  </span>
);

export default Logo;
