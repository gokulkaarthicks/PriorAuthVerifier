import { useCallback, useEffect, useRef, useState } from "react";
import { listJobs } from "../api/jobs";
import type { JobList, JobSummary } from "../types/jobs";

// One non-overlapping poller for the whole app, including while reviewing a case.
export function useJobQueue() {
  const [data, setData] = useState<JobList | null>(null);
  const [error, setError] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const controller = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const refresh = useCallback(async function refreshQueue() {
    clearTimeout(timer.current);
    controller.current?.abort();
    const current = ++generation.current;
    const abort = new AbortController();
    controller.current = abort;
    try {
      const next = await listJobs(abort.signal);
      if (current !== generation.current || abort.signal.aborted) return;
      setData(next);
      setError("");
      if (next.active_count > 0) timer.current = setTimeout(refreshQueue, 2500);
    } catch {
      if (current !== generation.current || abort.signal.aborted) return;
      setError("Case list unavailable. Reconnecting…");
      timer.current = setTimeout(refreshQueue, 5000);
    }
  }, []);
  useEffect(() => {
    void refresh();
    const onFocus = () => {
      void refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => {
      generation.current++;
      clearTimeout(timer.current);
      controller.current?.abort();
      window.removeEventListener("focus", onFocus);
    };
  }, [refresh]);
  function accepted(job: JobSummary) {
    setData((previous) =>
      previous
        ? {
            ...previous,
            jobs: [
              job,
              ...previous.jobs.filter((item) => item.job_id !== job.job_id),
            ],
            active_count:
              previous.active_count +
              (previous.jobs.some((item) => item.job_id === job.job_id)
                ? 0
                : 1),
          }
        : previous,
    );
    void refresh();
  }
  return { data, error, refresh, accepted };
}
