import { useEffect, useState } from "react";
const messages = [
  "Identifying the planned procedure",
  "Locating governing policy criteria",
  "Evaluating clinical documentation",
  "Checking authorization readiness",
];
export default function AnalysisLoading({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const update = () =>
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);
  return (
    <section className="loading-panel" aria-label="Analysis in progress">
      <div className="loading-heading">
        <span className="spinner" aria-hidden="true" />
        <h2>Analyzing authorization</h2>
        <span className="elapsed">
          {String(Math.floor(elapsed / 60)).padStart(2, "0")}:
          {String(elapsed % 60).padStart(2, "0")} elapsed
        </span>
      </div>
      <div
        className="indeterminate"
        role="progressbar"
        aria-label="Analyzing authorization"
      >
        <span />
      </div>
      <p>{messages[Math.floor(elapsed / 18) % messages.length]}</p>
      <p className="muted small">
        Analysis includes these checks; live stage progress is not available.
      </p>
      {elapsed >= 60 && (
        <p role="status">
          Large policy documents can take several minutes to evaluate.
        </p>
      )}
    </section>
  );
}
