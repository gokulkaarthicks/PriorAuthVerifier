import type { AnalysisResponse, ReadinessStatus } from "./analysis";
export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
export interface JobSummary {
  job_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  patient_filename: string;
  policy_filename: string;
  procedure_name: string | null;
  cpt_code: string | null;
  overall_status: ReadinessStatus | null;
  error_message: string | null;
}
export interface JobDetail extends JobSummary {
  result: AnalysisResponse | null;
}
export interface JobList {
  jobs: JobSummary[];
  active_count: number;
  max_active_jobs: number;
  execution_concurrency: number;
}
