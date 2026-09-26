import type { AnalysisResponse } from "../types/analysis";

export class AnalysisError extends Error {}

export function validateFile(file: File): string | null {
  const allowedTypes = [
    "",
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "application/vnd.pdf",
    "text/pdf",
    "application/octet-stream",
  ];
  if (
    !file.name.toLowerCase().endsWith(".pdf") ||
    !allowedTypes.includes(file.type)
  ) {
    return "Choose a PDF document. Other file types are not supported.";
  }
  if (file.size === 0)
    return "This PDF is empty. Choose a document with content.";
  return null;
}

// Validate the fields used in the UI before accepting a response at the boundary.
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
function nullableString(value: unknown): boolean {
  return value === null || typeof value === "string";
}
function evaluation(value: unknown): boolean {
  return (
    record(value) &&
    ["MET", "MISSING", "UNCLEAR"].includes(String(value.status)) &&
    typeof value.explanation === "string" &&
    Array.isArray(value.temporal_warnings) &&
    value.temporal_warnings.every((item) => typeof item === "string") &&
    Array.isArray(value.evidence) &&
    value.evidence.every(
      (item) =>
        record(item) &&
        typeof item.text === "string" &&
        typeof item.chunk_id === "string" &&
        (item.page_number === null ||
          (Number.isInteger(item.page_number) &&
            Number(item.page_number) > 0)) &&
        nullableString(item.document_date),
    )
  );
}
export function isAnalysis(value: unknown): value is AnalysisResponse {
  return (
    record(value) &&
    record(value.procedure) &&
    nullableString(value.procedure.procedure_name) &&
    nullableString(value.procedure.cpt_code) &&
    ["READY", "NOT_READY", "NEEDS_REVIEW", "NOT_APPLICABLE"].includes(
      String(value.overall_status),
    ) &&
    typeof value.matching_policy_criteria_found === "boolean" &&
    typeof value.next_step === "string" &&
    Array.isArray(value.criteria) &&
    value.criteria.every(
      (item) =>
        record(item) &&
        evaluation(item) &&
        typeof item.criterion_id === "string" &&
        nullableString(item.description),
    ) &&
    Array.isArray(value.exclusions) &&
    value.exclusions.every(
      (item) =>
        record(item) &&
        evaluation(item) &&
        typeof item.exclusion_id === "string" &&
        typeof item.triggered === "boolean",
    ) &&
    Array.isArray(value.missing_items) &&
    value.missing_items.every(
      (item) =>
        record(item) &&
        typeof item.criterion_id === "string" &&
        typeof item.description === "string",
    ) &&
    Array.isArray(value.review_items) &&
    value.review_items.every(
      (item) =>
        record(item) &&
        typeof item.criterion_id === "string" &&
        typeof item.description === "string" &&
        typeof item.explanation === "string",
    )
  );
}

export async function analyzeAuthorization(
  patientFile: File,
  policyFile: File,
): Promise<AnalysisResponse> {
  for (const file of [patientFile, policyFile]) {
    const error = validateFile(file);
    if (error) throw new AnalysisError(error);
  }
  const body = new FormData();
  body.append("patient_file", patientFile);
  body.append("policy_file", policyFile);
  let response: Response;
  try {
    // Intentionally no deadline and no automatic retries for long policy analyses.
    response = await fetch("/api/analyze", { method: "POST", body });
  } catch {
    throw new AnalysisError(
      "Connection lost. Check that the backend is running before trying again. The previous analysis may still be running.",
    );
  }
  if (!response.ok) {
    // Never display raw backend/provider messages, which can contain internal details.
    if (response.status === 413)
      throw new AnalysisError(
        "A document exceeds the server upload limit. Choose a smaller PDF and try again.",
      );
    if (response.status === 400 || response.status === 422)
      throw new AnalysisError(
        "The documents could not be validated or evaluated. Check that both PDFs are readable and contain the intended chart and policy.",
      );
    if (response.status === 502 || response.status === 503)
      throw new AnalysisError(
        "The analysis service is temporarily unavailable. Please try again when the service is available.",
      );
    throw new AnalysisError(
      "The server could not complete this analysis. Please try again.",
    );
  }
  try {
    const data: unknown = await response.json();
    if (isAnalysis(data)) return data;
  } catch {
    /* Treat malformed JSON as an unusable response, without exposing it. */
  }
  throw new AnalysisError(
    "The analysis returned an unreadable result. Please try again or contact your administrator.",
  );
}
