import type { AnalysisResponse } from "../src/types/analysis";
const criterion = {
  criterion_id: "internal-criterion-id",
  description: "Heart rate below 50 bpm",
  status: "MET" as const,
  explanation: "44 bpm documented on ECG",
  evidence: [
    {
      chunk_id: "internal-chunk-id",
      text: "Heart rate 44 bpm",
      page_number: 5,
      document_date: "2025-10-08",
    },
  ],
  temporal_warnings: [],
  evidence_chunk_ids: [],
  evidence_text: [],
  page_numbers: [],
};
export const result: AnalysisResponse = {
  procedure: {
    procedure_name: "Permanent pacemaker implantation",
    cpt_code: "33208",
    status: "planned",
    source_chunk_id: null,
    source_text: null,
    page_number: 4,
    is_future_procedure: true,
    confidence: 0.99,
  },
  matching_policy_criteria_found: true,
  overall_status: "NOT_READY",
  criteria: [criterion],
  exclusions: [
    {
      ...criterion,
      exclusion_id: "internal-exclusion-id",
      triggered: true,
      explanation: "Asymptomatic sinus node dysfunction",
      evidence: [
        {
          chunk_id: "private-id",
          text: "Patient remains fully asymptomatic with no functional limitation.",
          page_number: 4,
          document_date: "2025-11-12",
        },
      ],
    },
    {
      ...criterion,
      exclusion_id: "other-id",
      triggered: false,
      status: "MISSING",
      explanation: "Non-triggered policy check",
    },
  ],
  missing_items: [
    { criterion_id: "missing-id", description: "Syncope is not documented." },
  ],
  review_items: [],
  next_step:
    "Manual review is required because a policy non-indication is documented.",
};
