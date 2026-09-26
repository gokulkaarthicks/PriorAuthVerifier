import { useEffect, useRef, useState } from "react";
import { AnalysisError } from "./api/analyze";
import { createJob, getJob } from "./api/jobs";
import UploadPanel from "./components/UploadPanel";
import ReviewWorkspace from "./components/ReviewWorkspace";
import CaseQueue from "./components/CaseQueue";
import { useJobQueue } from "./hooks/useJobQueue";
import type { JobDetail, JobSummary } from "./types/jobs";

export default function App() {
  const [patient, setPatient] = useState<File | null>(null);
  const [policy, setPolicy] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [review, setReview] = useState<JobDetail | null>(null);
  const [opening, setOpening] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const submitLock = useRef(false);
  const detailRequest = useRef<AbortController | null>(null);
  const main = useRef<HTMLElement>(null);
  const queue = useJobQueue();
  const atCapacity =
    !!queue.data && queue.data.active_count >= queue.data.max_active_jobs;
  useEffect(() => () => detailRequest.current?.abort(), []);
  function focusMain() {
    window.requestAnimationFrame(() => {
      main.current?.focus();
      window.scrollTo({ top: 0 });
    });
  }
  async function submit(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitLock.current || atCapacity) return;
    if (!patient || !policy) {
      setError("Select a patient chart and policy PDF.");
      return;
    }
    submitLock.current = true;
    setSubmitting(true);
    setError("");
    try {
      const job = await createJob(patient, policy);
      queue.accepted(job);
      setPatient(null);
      setPolicy(null);
      setAnnouncement(
        "Case submitted. You can start another prior authorization.",
      );
    } catch (cause) {
      setError(
        cause instanceof AnalysisError
          ? cause.message
          : "Unable to submit this case.",
      );
      // Refresh after an ambiguous POST failure or capacity race. Never retry POST.
      void queue.refresh();
    } finally {
      submitLock.current = false;
      setSubmitting(false);
    }
  }
  async function openReview(job: JobSummary) {
    if (opening) return;
    const abort = new AbortController();
    detailRequest.current = abort;
    setOpening(job.job_id);
    setError("");
    try {
      const detail = await getJob(job.job_id, abort.signal);
      if (abort.signal.aborted) return;
      if (detail.status !== "COMPLETED" || !detail.result)
        throw new AnalysisError("This case is not ready for review.");
      setReview(detail);
      setAnnouncement(
        `Prior authorization review: ${detail.overall_status?.replaceAll("_", " ")}.`,
      );
      focusMain();
    } catch (cause) {
      if (!abort.signal.aborted)
        setError(
          cause instanceof AnalysisError
            ? cause.message
            : "Unable to open this case.",
        );
    } finally {
      setOpening(null);
    }
  }
  function backToCases() {
    setReview(null);
    setError("");
    void queue.refresh();
    focusMain();
  }
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="app-header">
        <div className="header-inner">
          <a
            className="brand"
            href="/"
            onClick={(event) => {
              event.preventDefault();
              backToCases();
            }}
            aria-label="PriorAuthVerifier home"
          >
            <span className="brand-mark" aria-hidden="true">
              P<span>✓</span>
            </span>
            <span>
              PriorAuthVerifier<small>Prior Authorization</small>
            </span>
          </a>
          <span className="header-note">
            {queue.data ? `${queue.data.active_count} active` : ""}
          </span>
        </div>
      </header>
      <div className="sr-only" role="status" aria-live="polite">
        {announcement}
      </div>
      <main id="main" tabIndex={-1} ref={main}>
        {review?.result ? (
          <ReviewWorkspace
            result={review.result}
            patientName={review.patient_filename}
            policyName={review.policy_filename}
            onReset={backToCases}
          />
        ) : (
          <>
            <div className="page-heading">
              <h1>New prior authorization</h1>
              <p>Upload the patient chart and policy.</p>
            </div>
            <form onSubmit={submit} aria-busy={submitting}>
              <div className="upload-grid">
                <UploadPanel
                  title="Patient chart"
                  number="01"
                  file={patient}
                  disabled={submitting}
                  onChange={(file) => {
                    setPatient(file);
                    setError("");
                  }}
                />
                <UploadPanel
                  title="Policy / guideline"
                  number="02"
                  file={policy}
                  disabled={submitting}
                  onChange={(file) => {
                    setPolicy(file);
                    setError("");
                  }}
                />
              </div>
              <div className="action-bar">
                <div>
                  {atCapacity && (
                    <p className="capacity-notice" role="status">
                      <strong>
                        {queue.data?.active_count} active prior authorizations
                      </strong>
                      <span>
                        Wait for a case to finish before starting another.
                      </span>
                    </p>
                  )}
                </div>
                <button
                  className="primary"
                  disabled={
                    !patient ||
                    !policy ||
                    submitting ||
                    atCapacity ||
                    !queue.data ||
                    !!queue.error
                  }
                  type="submit"
                >
                  {submitting ? "Submitting…" : "Start verification"}
                  <span aria-hidden="true">→</span>
                </button>
              </div>
            </form>
            {error && (
              <div className="error-panel" role="alert">
                {error}
              </div>
            )}
            {queue.error && (
              <div className="error-panel" role="status">
                {queue.error}{" "}
                <button
                  className="text-button"
                  onClick={() => void queue.refresh()}
                >
                  Refresh
                </button>
              </div>
            )}
            <CaseQueue
              data={queue.data}
              opening={opening}
              onReview={(job) => void openReview(job)}
            />
          </>
        )}
      </main>
    </>
  );
}
