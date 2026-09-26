import type {
  AnalysisResponse,
  CriterionEvaluation,
  EvidenceCitation,
  ExclusionEvaluation,
} from "../types/analysis";

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`status-badge status-${status.toLowerCase()}`}>
      {status.replaceAll("_", " ")}
    </span>
  );
}
function formatDate(value: string) {
  // Preserve date-only values without shifting the source date across time zones.
  const date = new Date(
    /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T12:00:00` : value,
  );
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      }).format(date);
}
function Evidence({
  citations,
  warnings,
}: {
  citations: EvidenceCitation[];
  warnings: string[];
}) {
  return (
    <>
      {citations.map((citation, index) => (
        <blockquote key={index}>
          <p>“{citation.text}”</p>
          <footer>
            Patient Chart
            {citation.page_number !== null && ` · Page ${citation.page_number}`}
            {citation.document_date &&
              ` · ${formatDate(citation.document_date)}`}
          </footer>
        </blockquote>
      ))}
      {warnings.map((warning, index) => (
        <p className="warning-text" key={index}>
          Timing note: {warning}
        </p>
      ))}
    </>
  );
}
function Exclusion({ item }: { item: ExclusionEvaluation }) {
  return (
    <article className="finding">
      <div className="finding-label">
        {item.triggered ? "Triggered non-indication" : "Policy check"}
        {!item.triggered && <StatusBadge status={item.status} />}
      </div>
      <h3>{item.explanation || "Policy non-indication"}</h3>
      <Evidence citations={item.evidence} warnings={item.temporal_warnings} />
    </article>
  );
}
function ClinicalEvidence({ criteria }: { criteria: CriterionEvaluation[] }) {
  return (
    <section className="surface">
      <div className="section-heading">
        <h2>Clinical Evidence</h2>
        <span className="count">{criteria.length} evaluated</span>
      </div>
      {criteria.length === 0 ? (
        <p className="empty-copy">
          No clinical evidence evaluations were returned.
        </p>
      ) : (
        criteria.map((item) => (
          <article className="evidence-row" key={item.criterion_id}>
            <div>
              <StatusBadge status={item.status} />
            </div>
            <div>
              <h3>{item.description || "Policy requirement"}</h3>
              <p className="explanation">
                {item.explanation ||
                  (item.status === "MISSING"
                    ? "Not documented"
                    : "No explanation provided.")}
              </p>
              <Evidence
                citations={item.evidence}
                warnings={item.temporal_warnings}
              />
            </div>
          </article>
        ))
      )}
    </section>
  );
}
function MissingRequirements({ result }: { result: AnalysisResponse }) {
  return (
    <section className="surface">
      <div className="section-heading">
        <h2>Missing or unclear documentation</h2>
      </div>
      <div className="section-body">
        {result.missing_items.length === 0 &&
          result.review_items.length === 0 && (
            <p className="muted">
              No missing or review items were returned by the analysis.
            </p>
          )}
        {result.missing_items.map((item, index) => (
          <div className="requirement" key={`missing-${index}`}>
            <StatusBadge status="MISSING" />
            <p>{item.description}</p>
          </div>
        ))}
        {result.review_items.map((item, index) => (
          <div className="requirement" key={`review-${index}`}>
            <span className="status-badge status-unclear">REVIEW</span>
            <p>{item.description}</p>
            {item.explanation !== item.description && (
              <p className="muted">{item.explanation}</p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
const outcomeCopy = {
  READY:
    "Documentation satisfies the evaluated requirements of the uploaded policy.",
  NOT_READY:
    "Review the blocking findings and documentation requirements below before submission.",
  NEEDS_REVIEW:
    "The analysis requires manual review. Review the evidence and flagged items below.",
  NOT_APPLICABLE:
    "The uploaded policy does not contain criteria for the detected planned procedure.",
};
export default function ReviewWorkspace({
  result,
  patientName,
  policyName,
  onReset,
}: {
  result: AnalysisResponse;
  patientName: string;
  policyName: string;
  onReset: () => void;
}) {
  const triggered = result.exclusions.filter((item) => item.triggered);
  const otherChecks = result.exclusions.filter((item) => !item.triggered);
  return (
    <>
      <div className="page-heading review-heading">
        <div>
          <p className="eyebrow">AUTHORIZATION REVIEW</p>
          <h1>Verification results</h1>
          <p>Evidence and requirements from the uploaded documents.</p>
        </div>
        <button className="secondary" onClick={onReset}>
          New verification <span aria-hidden="true">↗</span>
        </button>
      </div>
      <section
        className="surface result-overview"
        aria-label="Analysis summary"
      >
        <dl className="metadata">
          <div>
            <dt>Procedure</dt>
            <dd>{result.procedure.procedure_name || "Not identified"}</dd>
          </div>
          <div>
            <dt>CPT</dt>
            <dd>{result.procedure.cpt_code || "Not identified"}</dd>
          </div>
          <div>
            <dt>Policy</dt>
            <dd>
              {result.matching_policy_criteria_found
                ? "Matching criteria found"
                : "No matching criteria"}
            </dd>
          </div>
        </dl>
        <div
          className={`outcome outcome-${result.overall_status.toLowerCase()}`}
        >
          <StatusBadge status={result.overall_status} />
          {result.overall_status === "NOT_APPLICABLE" && (
            <h2>No matching procedure criteria</h2>
          )}
          <p>{outcomeCopy[result.overall_status]}</p>
        </div>
        <div className="source-files">
          <span>
            <strong>Patient Chart</strong> {patientName}
          </span>
          <span>
            <strong>Policy / Guideline</strong> {policyName}
          </span>
        </div>
      </section>
      {triggered.length > 0 && (
        <section className="surface blocking">
          <div className="section-heading">
            <h2>Blocking policy findings</h2>
            <span className="count">{triggered.length} triggered</span>
          </div>
          <div className="section-body">
            {triggered.map((item) => (
              <Exclusion key={item.exclusion_id} item={item} />
            ))}
          </div>
        </section>
      )}
      <div className="review-grid">
        <div className="review-main">
          <ClinicalEvidence criteria={result.criteria} />
          <details className="surface disclosure">
            <summary>
              All policy criteria{" "}
              <span className="count">{result.criteria.length}</span>
            </summary>
            <div className="section-body">
              <p className="muted small">
                Evaluated conditions from the uploaded policy.
              </p>
              {result.criteria.map((item) => (
                <div className="policy-condition" key={item.criterion_id}>
                  <span className="eyebrow">CONDITION</span>
                  <p>{item.description || item.explanation}</p>
                  <StatusBadge status={item.status} />
                </div>
              ))}
            </div>
          </details>
          {otherChecks.length > 0 && (
            <details className="surface disclosure">
              <summary>
                View all policy checks{" "}
                <span className="count">
                  {otherChecks.length} not triggered
                </span>
              </summary>
              <div className="section-body">
                {otherChecks.map((item) => (
                  <Exclusion key={item.exclusion_id} item={item} />
                ))}
              </div>
            </details>
          )}
        </div>
        <aside className="review-aside">
          <section className="surface next-step">
            <span className="eyebrow">FOLLOW-UP</span>
            <h2>Recommended next step</h2>
            <p>
              {result.next_step || "No next step was provided by the analysis."}
            </p>
          </section>
          <MissingRequirements result={result} />
        </aside>
      </div>
      <p className="disclaimer">
        This review evaluates documentation against the uploaded policy and does
        not guarantee payer authorization.
      </p>
    </>
  );
}
