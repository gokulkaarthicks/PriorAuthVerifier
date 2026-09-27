import { useEffect, useState } from "react";
import type { JobList, JobSummary } from "../types/jobs";
import { StatusBadge } from "./ReviewWorkspace";

function elapsed(started: string | null, now: number) {
  if (!started) return "";
  const seconds = Math.max(0, Math.floor((now - Date.parse(started)) / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
export default function CaseQueue({
  data,
  opening,
  onReview,
}: {
  data: JobList | null;
  opening: string | null;
  onReview: (job: JobSummary) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const [now, setNow] = useState(Date.now());
  const [errorId, setErrorId] = useState<string | null>(null);
  const active = data?.active_count ?? 0;
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return (
    <section className="case-queue" aria-labelledby="cases-heading">
      <div className="queue-heading">
        <h2 id="cases-heading">
          Prior Authorization cases{" "}
          <span className="count">{data?.jobs.length ?? 0}</span>
        </h2>
        <span className="capacity" role="status">
          {data
            ? `${active} active / ${data.max_active_jobs}`
            : "Loading cases…"}
        </span>
      </div>
      <div className="surface">
        <button
          type="button"
          className="queue-disclosure"
          aria-expanded={expanded}
          aria-controls="case-list"
          onClick={() => setExpanded((value) => !value)}
        >
          <span aria-hidden="true">{expanded ? "⌄" : "›"}</span> Recent cases
        </button>
        <div id="case-list" hidden={!expanded}>
          {!data?.jobs.length ? (
            <p className="empty-copy">
              {data ? "No cases in this session." : "Loading…"}
            </p>
          ) : (
            <div className="case-table-scroll">
              <table className="case-table">
                <thead>
                  <tr>
                    <th scope="col">Patient / Procedure</th>
                    <th scope="col">Policy</th>
                    <th scope="col">Status</th>
                    <th scope="col">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {data.jobs.map((job) => (
                    <tr key={job.job_id}>
                      <td data-label="Patient">
                        <strong className="case-filename">
                          {job.patient_filename}
                        </strong>
                        {job.procedure_name && (
                          <span className="case-procedure">
                            {job.procedure_name}
                            {job.cpt_code && ` · ${job.cpt_code}`}
                          </span>
                        )}
                      </td>
                      <td data-label="Policy">{job.policy_filename}</td>
                      <td data-label="Status">
                        {job.status === "COMPLETED" && job.overall_status ? (
                          <StatusBadge status={job.overall_status} />
                        ) : job.status === "RUNNING" ? (
                          <span className="job-running">
                            Analyzing{" "}
                            <span className="elapsed">
                              · {elapsed(job.started_at, now)}
                            </span>
                          </span>
                        ) : job.status === "QUEUED" ? (
                          <span className="status-badge status-unclear">
                            Queued
                          </span>
                        ) : (
                          <span className="status-badge">Failed</span>
                        )}
                      </td>
                      <td data-label="Action">
                        {job.status === "COMPLETED" ? (
                          <button
                            className="text-button"
                            disabled={opening !== null}
                            onClick={() => onReview(job)}
                            aria-label={`Review ${job.patient_filename}`}
                          >
                            {opening === job.job_id ? "Opening…" : "Review"}
                          </button>
                        ) : job.status === "FAILED" ? (
                          <>
                            <button
                              className="text-button"
                              aria-expanded={errorId === job.job_id}
                              onClick={() =>
                                setErrorId(
                                  errorId === job.job_id ? null : job.job_id,
                                )
                              }
                            >
                              Details
                            </button>
                            {errorId === job.job_id && (
                              <p className="job-error">
                                Analysis could not be completed.
                              </p>
                            )}
                          </>
                        ) : (
                          <span className="muted">-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
