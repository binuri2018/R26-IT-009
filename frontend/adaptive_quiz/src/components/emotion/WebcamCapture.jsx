/**
 * components/emotion/WebcamCapture.jsx
 * Shows a small live webcam feed in the corner during assessment.
 * Displays the current detected expression as a badge.
 * The video element is bound to the videoRef from useWebcam hook.
 */
import { getExpressionEmoji } from '../../utils/quizLogic';
import './WebcamCapture.css';

const WebcamCapture = ({ videoRef, currentExpression, isWebcamActive, error }) => {
  return (
    <div className="webcam-panel">
      <div className="webcam-header">
        <span className={`webcam-status-dot ${isWebcamActive ? 'active' : 'inactive'}`} />
        <span className="webcam-label">Face Tracking</span>
      </div>

      {/* Live video feed */}
      <div className="webcam-video-wrapper">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="webcam-video"
        />
        {!isWebcamActive && (
          <div className="webcam-offline">
            <span>📷</span>
            <span>{error ? 'Camera unavailable' : 'Starting…'}</span>
          </div>
        )}
      </div>

      {/* Expression badge */}
      <div className="webcam-expression">
        <span className="expression-emoji">{getExpressionEmoji(currentExpression)}</span>
        <span className="expression-label">
          {currentExpression === 'unknown' ? 'no face detected' : currentExpression}
        </span>
      </div>
    </div>
  );
};

export default WebcamCapture;
