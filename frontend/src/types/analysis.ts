// Mirrors app/schemas/{analysis,evaluation,procedure,policy}.py. No inferred decisions.
export type CriterionStatus = "MET" | "MISSING" | "UNCLEAR";
export type DisplayStatus =
  | "MET"
  | "MISSING"
  | "UNCLEAR"
  | "NOT_SATISFIED";
export type ReadinessStatus =
  "READY" | "NOT_READY" | "NEEDS_REVIEW" | "NOT_APPLICABLE";
export type CriterionNodeType = "CONDITION" | "AND" | "OR";
export interface EvidenceCitation {
  chunk_id: string;
  text: string;
  page_number: number | null;
  document_date: string | null;
}
interface Evaluation {
  status: CriterionStatus;
  evidence: EvidenceCitation[];
  explanation: string;
  temporal_warnings: string[];
  evidence_chunk_ids: string[];
  evidence_text: string[];
  page_numbers: number[];
}
export interface CriterionEvaluation extends Evaluation {
  criterion_id: string;
  description: string | null;
}
export interface ExclusionEvaluation extends Evaluation {
  exclusion_id: string;
  triggered: boolean;
}
export interface CriterionNode {
  id: string;
  node_type: CriterionNodeType;
  description: string | null;
  category?: string | null;
  temporal_requirement?: string | null;
  children: CriterionNode[];
  source_chunk_id?: string | null;
  source_text?: string | null;
  page_number?: number | null;
}
export interface ProcedureExtraction {
  procedure_name: string | null;
  cpt_code: string | null;
  status: string | null;
  source_chunk_id: string | null;
  source_text: string | null;
  page_number: number | null;
  is_future_procedure: boolean;
  confidence: number;
}
export interface ActionableGap {
  criterion_id: string;
  description: string;
  detail?: string;
}
export interface AnalysisResponse {
  procedure: ProcedureExtraction;
  matching_policy_criteria_found: boolean;
  overall_status: ReadinessStatus;
  criteria: CriterionEvaluation[];
  exclusions: ExclusionEvaluation[];
  missing_items: { criterion_id: string; description: string }[];
  review_items: {
    criterion_id: string;
    description: string;
    explanation: string;
  }[];
  next_step: string;
  criteria_tree?: CriterionNode | null;
}
