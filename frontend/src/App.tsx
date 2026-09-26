import { useRef, useState } from "react";
import { AnalysisError, analyzeAuthorization } from "./api/analyze";
import UploadPanel from "./components/UploadPanel";
import AnalysisLoading from "./components/AnalysisLoading";
import ReviewWorkspace from "./components/ReviewWorkspace";
import type { AnalysisResponse } from "./types/analysis";

export default function App() {
  const [patient, setPatient] = useState<File | null>(null);
  const [policy, setPolicy] = useState<File | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const submitting = useRef(false);
  const resultFocus = useRef<HTMLDivElement>(null);
  const busy = startedAt !== null;
  async function run(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting.current) return;
    if (!patient || !policy) {
      setError("Select both a patient chart PDF and a policy PDF.");
      return;
    }
    submitting.current = true;
    setError("");
    setStartedAt(Date.now());
    setAnnouncement("Analyzing authorization. This may take several minutes.");
    try {
      const response = await analyzeAuthorization(patient, policy);
      setResult(response);
      setAnnouncement(
        `Analysis complete: ${response.overall_status.replaceAll("_", " ")}.`,
      );
      window.requestAnimationFrame(() => {
        resultFocus.current?.focus();
        window.scrollTo({ top: 0 });
      });
    } catch (cause) {
      setError(
        cause instanceof AnalysisError
          ? cause.message
          : "Analysis could not be completed. Please try again.",
      );
      setAnnouncement("Analysis could not be completed.");
    } finally {
      submitting.current = false;
      setStartedAt(null);
    }
  }
  function reset() {
    setResult(null);
    setPatient(null);
    setPolicy(null);
    setError("");
    setAnnouncement("Ready for a new verification.");
    window.requestAnimationFrame(() => resultFocus.current?.focus());
  }
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="app-header">
        <div className="header-inner">
          <a className="brand" href="/" aria-label="PriorAuthVerifier home">
            <span className="brand-mark" aria-hidden="true">
              P<span>✓</span>
            </span>
            <span>
              PriorAuthVerifier<small>Authorization Review</small>
            </span>
          </a>
          <span className="header-note">
            <span aria-hidden="true" className="grounded-dot" />
            Evidence-grounded review
          </span>
        </div>
      </header>
      <div className="sr-only" role="status" aria-live="polite">
        {announcement}
      </div>
      <main id="main" tabIndex={-1} ref={resultFocus}>
        {result ? (
          <ReviewWorkspace
            result={result}
            patientName={patient?.name ?? ""}
            policyName={policy?.name ?? ""}
            onReset={reset}
          />
        ) : (
          <>
            <div className="page-heading">
              <p className="eyebrow">DOCUMENT VERIFICATION</p>
              <h1>Verify authorization readiness</h1>
              <p>
                Compare a patient chart against the uploaded medical-necessity
                policy.
              </p>
            </div>
            <form onSubmit={run} aria-busy={busy}>
              <div className="upload-grid">
                <UploadPanel
                  number="01"
                  title="Patient Chart"
                  subtitle="Clinical notes and supporting documentation"
                  file={patient}
                  disabled={busy}
                  onChange={(file) => {
                    setPatient(file);
                    setError("");
                  }}
                />
                <UploadPanel
                  number="02"
                  title="Policy / Guideline"
                  subtitle="Medical-necessity criteria for the procedure"
                  file={policy}
                  disabled={busy}
                  onChange={(file) => {
                    setPolicy(file);
                    setError("");
                  }}
                />
              </div>
              <div className="action-bar">
                <p className="privacy-note">
                  <span aria-hidden="true">◇</span> Files are processed for this
                  analysis and are not stored by the browser.
                </p>
                <button
                  className="primary"
                  type="submit"
                  disabled={!patient || !policy || busy}
                >
                  {busy ? "Verification in progress" : "Run verification"}
                  <span aria-hidden="true">→</span>
                </button>
              </div>
              {error && (
                <div className="error-panel" role="alert">
                  <strong>Unable to complete verification</strong>
                  <p>{error}</p>
                  <span className="small">
                    Your selected files are still available. Run verification to
                    try again.
                  </span>
                </div>
              )}
              {startedAt !== null && <AnalysisLoading startedAt={startedAt} />}
            </form>
            {!busy && (
              <section className="workflow-note">
                <span className="eyebrow">WHAT THIS REVIEW INCLUDES</span>
                <div>
                  <p>
                    <strong>Procedure identification</strong>
                    <span>Find the planned procedure in the chart.</span>
                  </p>
                  <p>
                    <strong>Policy comparison</strong>
                    <span>Evaluate documentation against policy criteria.</span>
                  </p>
                  <p>
                    <strong>Readiness review</strong>
                    <span>Surface findings, evidence, and next steps.</span>
                  </p>
                </div>
              </section>
            )}
          </>
        )}
      </main>
      <footer className="app-footer">
        <span>PriorAuthVerifier</span>
        <span>Documentation review · Clinical judgment remains essential</span>
      </footer>
    </>
  );
}
