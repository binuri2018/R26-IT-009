/**
 * components/brand/LottiePlayer.jsx
 * Minimal Lottie player (lottie-web light build, SVG renderer).
 * With prefers-reduced-motion it shows a single still frame instead of looping.
 */
import { useEffect, useRef } from 'react';
import lottie from 'lottie-web/build/player/lottie_light';

const LottiePlayer = ({ animationData, className = '', loop = true, stillFrame = 110 }) => {
  const ref = useRef(null);

  useEffect(() => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const anim = lottie.loadAnimation({
      container: ref.current,
      renderer: 'svg',
      loop,
      autoplay: !reduce,
      animationData,
      rendererSettings: { preserveAspectRatio: 'xMidYMid meet' },
    });
    if (reduce) anim.goToAndStop(stillFrame, true);
    return () => anim.destroy();
  }, [animationData, loop, stillFrame]);

  return <div ref={ref} className={className} aria-hidden="true" />;
};

export default LottiePlayer;
