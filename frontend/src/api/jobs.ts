import { AnalysisError, isAnalysis, validateFile } from "./analyze";
import type { JobDetail, JobList, JobSummary } from "../types/jobs";

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
function nullableText(value: unknown) {
  return value === null || typeof value === "string";
}
function timestamp(value: unknown) {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}
function isJob(value: unknown): value is JobSummary {
  return (
    record(value) &&
    typeof value.job_id === "string" &&
    ["QUEUED", "RUNNING", "COMPLETED", "FAILED"].includes(
      String(value.status),
    ) &&
    timestamp(value.created_at) &&
    (value.started_at === null || timestamp(value.started_at)) &&
    (value.completed_at === null || timestamp(value.completed_at)) &&
    typeof value.patient_filename === "string" &&
    typeof value.policy_filename === "string" &&
    nullableText(value.procedure_name) &&
    nullableText(value.cpt_code) &&
    nullableText(value.error_message) &&
    (value.overall_status === null ||
      ["READY", "NOT_READY", "NOT_APPLICABLE", "NEEDS_REVIEW"].includes(
        String(value.overall_status),
      )) &&
    (value.status !== "COMPLETED" || value.overall_status !== null)
  );
}
async function request(url: string, init?: RequestInit): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    if (init?.signal?.aborted) throw error;
    throw new AnalysisError(
      "Connection lost. Check the case list before submitting again.",
    );
  }
  if (!response.ok) {
    if (response.status === 429)
      throw new AnalysisError("Wait for an active case to finish.");
    if (response.status === 404)
      throw new AnalysisError(
        "Case not found. The server session may have restarted.",
      );
    if (response.status === 413)
      throw new AnalysisError(
        "A PDF exceeds the upload limit. Choose a smaller file.",
      );
    if (response.status === 400 || response.status === 422)
      throw new AnalysisError("Check that both files are readable PDFs.");
    throw new AnalysisError("The service is unavailable. Please try again.");
  }
  try {
    return await response.json();
  } catch {
    throw new AnalysisError("The service returned an unreadable response.");
  }
}
export async function createJob(
  patient: File,
  policy: File,
): Promise<JobSummary> {
  for (const file of [patient, policy]) {
    const error = validateFile(file);
    if (error) throw new AnalysisError(error);
  }
  const body = new FormData();
  body.append("patient_file", patient);
  body.append("policy_file", policy);
  const data = await request("/api/jobs", { method: "POST", body });
  if (!isJob(data))
    throw new AnalysisError(
      "The service returned an unreadable case. Check the case list before submitting again.",
    );
  return data;
}
export async function listJobs(signal?: AbortSignal): Promise<JobList> {
  const data = await request("/api/jobs", { signal, cache: "no-store" });
  if (
    !record(data) ||
    !Array.isArray(data.jobs) ||
    !data.jobs.every(isJob) ||
    !Number.isInteger(data.active_count) ||
    Number(data.active_count) < 0 ||
    !Number.isInteger(data.max_active_jobs) ||
    Number(data.max_active_jobs) < 1 ||
    !Number.isInteger(data.execution_concurrency) ||
    Number(data.execution_concurrency) < 1
  ) {
    throw new AnalysisError("The case list is unavailable. Please refresh.");
  }
  return data as unknown as JobList;
}
export async function getJob(
  id: string,
  signal?: AbortSignal,
): Promise<JobDetail> {
  const data = await request(`/api/jobs/${encodeURIComponent(id)}`, {
    signal,
    cache: "no-store",
  });
  if (
    !record(data) ||
    !isJob(data) ||
    !(data.result === null || isAnalysis(data.result)) ||
    (data.status === "COMPLETED" && data.result === null)
  ) {
    throw new AnalysisError("This case returned an unreadable result.");
  }
  return data as unknown as JobDetail;
}
