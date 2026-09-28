import type {
  AnalysisResponse,
  DisplayStatus,
  EvidenceCitation,
  ExclusionEvaluation,
} from "../types/analysis";
import {
  curateResultView,
  type PrimaryCriterion,
} from "../utils/curateResultView";

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  return (
    <span className={`status-badge status-${normalized}`}>
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
}: {
  citations: EvidenceCitation[];
}) {
  return (
    <>
      {citations.map((citation, index) => (
        <div className="citation" key={index}>
          <p className="citation-source">
            Patient chart
            {citation.page_number !== null && ` · Page ${citation.page_number}`}
            {citation.document_date &&
              ` · ${formatDate(citation.document_date)}`}
          </p>
          <details className="source-evidence">
            <summary>Show source evidence</summary>
            <blockquote>
              <p>“{citation.text}”</p>
            </blockquote>
          </details>
        </div>
      ))}
    </>
  );
}
function Exclusion({ item }: { item: ExclusionEvaluation }) {
  return (
    <article className="finding">
      <div className="finding-label">
        {item.triggered ? "Triggered non-indication" : "Policy check"}
        {!item.triggered && <StatusBadge status="NOT_TRIGGERED" />}
      </div>
      <h3>{item.explanation || "Policy non-indication"}</h3>
      <Evidence citations={item.evidence} />
    </article>
  );
}
function RelevantCriteria({
  criteria,
  sourceDateNotes,
}: {
  criteria: PrimaryCriterion[];
  sourceDateNotes: string[];
}) {
  return (
    <section className="surface">
      <div className="section-heading">
        <h2>Relevant policy criteria</h2>
        <span className="count">{criteria.length} shown</span>
      </div>
      {sourceDateNotes.length > 0 && (
        <div className="section-body source-date-notes">
          {sourceDateNotes.map((note, index) => (
            <p className="warning-text" key={index}>
              Source-date note: {note}
            </p>
          ))}
        </div>
      )}
      {criteria.length === 0 ? (
        <p className="empty-copy">
          No relevant policy criteria were selected for primary display.
        </p>
      ) : (
        criteria.map((item) => (
          <article className="evidence-row" key={item.criterion_id}>
            <div>
              <StatusBadge status={item.display_status as DisplayStatus} />
            </div>
            <div>
              <h3>{item.description || "Policy requirement"}</h3>
              <p className="explanation">
                {item.explanation ||
                  (item.status === "MISSING"
                    ? "Not documented"
                    : "No explanation provided.")}
              </p>
              <Evidence citations={item.evidence} />
            </div>
          </article>
        ))
      )}
    </section>
  );
}
function MissingRequirements({
  missingItems,
  reviewItems,
}: {
  missingItems: { criterion_id: string; description: string; detail?: string }[];
  reviewItems: AnalysisResponse["review_items"];
}) {
  return (
    <section className="surface">
      <div className="section-heading">
        <h2>Missing or unclear documentation</h2>
      </div>
      <div className="section-body">
        {missingItems.length === 0 && reviewItems.length === 0 && (
          <p className="muted">
            No missing or review items were returned by the analysis.
          </p>
        )}
        {missingItems.map((item, index) => (
          <div className="requirement" key={`missing-${index}`}>
            <StatusBadge
              status={
                /affirmatively not met/i.test(item.description)
                  ? "NOT_SATISFIED"
                  : "MISSING"
              }
            />
            <div>
              <p>{item.description}</p>
              {item.detail && <p className="muted">{item.detail}</p>}
            </div>
          </div>
        ))}
        {reviewItems.map((item, index) => {
          const explanation =
            item.explanation &&
            item.explanation !== item.description &&
            !item.description.includes(item.explanation)
              ? item.explanation
              : null;
          return (
            <div className="requirement" key={`review-${index}`}>
              <span className="status-badge status-unclear">REVIEW</span>
              <p>{item.description}</p>
              {explanation && <p className="muted">{explanation}</p>}
            </div>
          );
        })}
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
  const curated = curateResultView(result);
  const showRelevant =
    result.matching_policy_criteria_found &&
    result.overall_status !== "NOT_APPLICABLE";
  return (
    <>
      <div className="page-heading review-heading">
        <div>
          <button className="text-button back-button" onClick={onReset}>
            ← Back to Prior Authorization cases
          </button>
          <h1>Prior Authorization review</h1>
        </div>
        <button className="secondary" onClick={onReset}>
          New Prior Authorization <span aria-hidden="true">↗</span>
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
            <strong>Patient chart</strong> {patientName}
          </span>
          <span>
            <strong>Policy / guideline</strong> {policyName}
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
          {showRelevant && (
            <RelevantCriteria
              criteria={curated.primaryCriteria}
              sourceDateNotes={curated.sourceDateNotes}
            />
          )}
          {result.matching_policy_criteria_found && (
            <details className="surface disclosure">
              <summary>
                View all policy criteria{" "}
                <span className="count">{result.criteria.length}</span>
              </summary>
              <div className="section-body">
                <p className="muted small">
                  Complete evaluated conditions from the uploaded policy.
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
          )}
          {otherChecks.length > 0 && (
            <details className="surface disclosure">
              <summary>
                View non-triggered policy checks{" "}
                <span className="count">{otherChecks.length}</span>
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
            <h2>Recommended next step</h2>
            <p>
              {curated.nextStep ||
                result.next_step ||
                "No next step was provided by the analysis."}
            </p>
          </section>
          {result.matching_policy_criteria_found && (
            <MissingRequirements
              missingItems={curated.missingItems}
              reviewItems={curated.reviewItems}
            />
          )}
        </aside>
      </div>
      <p className="disclaimer">
        Policy review support only; final authorization decisions remain with
        the payer.
      </p>
    </>
  );
}
