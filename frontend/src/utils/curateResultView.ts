import type {
  ActionableGap,
  AnalysisResponse,
  CriterionEvaluation,
  CriterionNode,
  DisplayStatus,
} from "../types/analysis";

/** Affirmative-conflict cues already present in backend explanations/copy. */
const AFFIRMATIVE_TOKENS = [
  "affirmatively",
  "contradict",
  "disprov",
  "documented absence",
  "documented as absent",
  "not present",
  "rules out",
];

const CONSOLIDATED_SOURCE_DATE_NOTE =
  "The chart contains conflicting dates for some clinical records; extracted source dates were preserved without rewriting chronology.";

export type PrimaryCriterion = CriterionEvaluation & {
  display_status: DisplayStatus;
};

export interface CuratedResultView {
  primaryCriteria: PrimaryCriterion[];
  sourceDateNotes: string[];
  missingItems: ActionableGap[];
  reviewItems: AnalysisResponse["review_items"];
  nextStep: string;
}

/**
 * Presentation-only curation from already-evaluated API fields.
 * Unsupported alternative MISSING leaves stay in `result.criteria` for the
 * collapsed full list; they are omitted from the primary pathway view.
 */
export function curateResultView(result: AnalysisResponse): CuratedResultView {
  if (!result.matching_policy_criteria_found) {
    return {
      primaryCriteria: [],
      sourceDateNotes: [],
      missingItems: [],
      reviewItems: [],
      nextStep: result.next_step,
    };
  }

  const byId = new Map(
    result.criteria.map((item) => [item.criterion_id, item]),
  );
  const root = result.criteria_tree ?? null;
  const relevantIds = root
    ? selectRelevantConditionIds(root, byId)
    : new Set(result.criteria.map((item) => item.criterion_id));

  const exclusionIds = new Set(
    result.exclusions.map((item) => item.exclusion_id),
  );
  const triggeredIds = new Set(
    result.exclusions
      .filter((item) => item.triggered)
      .map((item) => item.exclusion_id),
  );

  // Primary missing panel: only gaps already returned for leaves on the
  // curated relevant pathway. Do not harvest extra MISSING rows from the
  // full criteria tree.
  const scopedMissing = result.missing_items.filter((item) =>
    relevantIds.has(item.criterion_id),
  );

  const missingItems = elevateMissingToParents(
    scopedMissing,
    root,
    result.criteria,
    relevantIds,
  );

  const actionableIds = new Set(missingItems.map((item) => item.criterion_id));
  for (const item of result.review_items) {
    if (
      !exclusionIds.has(item.criterion_id) &&
      relevantIds.has(item.criterion_id)
    ) {
      actionableIds.add(item.criterion_id);
    }
  }

  const elevatedLeafIds = new Set(
    missingItems.flatMap((item) => item.elevatedLeafIds ?? []),
  );

  let primary = result.criteria.filter((item) => {
    if (!relevantIds.has(item.criterion_id)) return false;
    if (elevatedLeafIds.has(item.criterion_id)) return false;
    if (actionableIds.has(item.criterion_id)) return true;
    if (item.status === "MET" || item.status === "UNCLEAR") {
      return item.evidence.length > 0;
    }
    return false;
  });

  primary = deprioritizeGenericMet(primary, {
    hasActionableGap: actionableIds.size > 0 || triggeredIds.size > 0,
  });

  const missingById = new Map(
    missingItems.map((item) => [item.criterion_id, item]),
  );

  const primaryRows: PrimaryCriterion[] = primary
    .map((item) => ({
      ...item,
      display_status: displayStatus(item, missingById.get(item.criterion_id)),
      temporal_warnings: [] as string[],
    }))
    .sort((left, right) => primarySortKey(left) - primarySortKey(right));

  for (const gap of missingItems) {
    if (primaryRows.some((row) => row.criterion_id === gap.criterion_id)) {
      continue;
    }
    if (!(gap.elevatedLeafIds && gap.elevatedLeafIds.length > 0)) {
      continue;
    }
    primaryRows.unshift({
      criterion_id: gap.criterion_id,
      status: "MISSING",
      display_status: "MISSING",
      description: shortRequirement(gap.description),
      explanation: gap.detail || gap.description,
      evidence: [],
      temporal_warnings: [],
      evidence_chunk_ids: [],
      evidence_text: [],
      page_numbers: [],
    });
  }
  primaryRows.sort(
    (left, right) => primarySortKey(left) - primarySortKey(right),
  );

  const triggered = result.exclusions.filter((item) => item.triggered);
  const hasTimingConflict =
    result.criteria.some(
      (item) =>
        relevantIds.has(item.criterion_id) && item.temporal_warnings.length > 0,
    ) || triggered.some((item) => item.temporal_warnings.length > 0);
  const sourceDateNotes = hasTimingConflict
    ? [CONSOLIDATED_SOURCE_DATE_NOTE]
    : [];

  const reviewItems = result.review_items.filter(
    (item) =>
      !triggeredIds.has(item.criterion_id) &&
      relevantIds.has(item.criterion_id),
  );

  const displayGaps: ActionableGap[] = missingItems.map(
    ({ criterion_id, description, detail }) => ({
      criterion_id,
      description,
      detail,
    }),
  );

  return {
    primaryCriteria: primaryRows,
    sourceDateNotes,
    missingItems: displayGaps,
    reviewItems,
    nextStep: composeNextStep(result, {
      primaryCriteria: primaryRows,
      missingItems: displayGaps,
      reviewItems,
      triggered,
    }),
  };
}

/**
 * Walk the policy tree and keep condition IDs from the relevant pathway.
 * Outside a chosen pathway, pick a single closest supported OR branch (do not
 * accumulate every alternative that shares overlapping MET/UNCLEAR leaves).
 * Inside a chosen pathway, keep every supported OR alternative and every unmet
 * nested OR child so parent-gap elevation can represent the whole requirement.
 *
 * Outside a chosen pathway, unsupported AND sibling groups are omitted so
 * primary missing items stay on the selected indication path.
 */
export function selectRelevantConditionIds(
  node: CriterionNode,
  byId: Map<string, CriterionEvaluation>,
  insideSupportedPathway = false,
): Set<string> {
  if (node.node_type === "CONDITION") {
    return new Set([node.id]);
  }

  if (node.node_type === "AND") {
    const ids = new Set<string>();
    const targets = andPathwayChildren(node, byId, insideSupportedPathway);
    for (const child of targets) {
      for (const id of selectRelevantConditionIds(
        child,
        byId,
        insideSupportedPathway,
      )) {
        ids.add(id);
      }
    }
    return ids;
  }

  const supported = node.children.filter((child) =>
    branchHasSupport(child, byId),
  );
  let targets: CriterionNode[];
  if (supported.length > 0) {
    if (insideSupportedPathway) {
      targets = supported;
    } else {
      const best = closestAmong(supported, byId);
      targets = best ? [best] : [];
    }
  } else if (insideSupportedPathway) {
    targets = [...node.children];
  } else {
    const closest = closestAmong(node.children, byId);
    targets = closest ? [closest] : [];
  }

  const ids = new Set<string>();
  for (const child of targets) {
    for (const id of selectRelevantConditionIds(child, byId, true)) {
      ids.add(id);
    }
  }
  return ids;
}

/**
 * AND children for primary pathway display.
 * On a chosen pathway, keep every conjunct. Outside one, keep direct condition
 * leaves plus supported composite groups only (skip empty alternative groups).
 */
function andPathwayChildren(
  node: CriterionNode,
  byId: Map<string, CriterionEvaluation>,
  insideSupportedPathway: boolean,
): CriterionNode[] {
  if (insideSupportedPathway) {
    return [...node.children];
  }

  const conditions: CriterionNode[] = [];
  const composites: CriterionNode[] = [];
  for (const child of node.children) {
    if (child.node_type === "CONDITION") conditions.push(child);
    else composites.push(child);
  }

  if (composites.length === 0) {
    return conditions;
  }

  const supported = composites.filter((child) => branchHasSupport(child, byId));
  if (supported.length > 0) {
    return [...conditions, ...supported];
  }

  const closest = closestAmong(composites, byId);
  return closest ? [...conditions, closest] : conditions;
}

function branchHasSupport(
  node: CriterionNode,
  byId: Map<string, CriterionEvaluation>,
): boolean {
  return collectConditionIds(node).some((id) => {
    const item = byId.get(id);
    if (!item) return false;
    if (item.status !== "MET" && item.status !== "UNCLEAR") return false;
    return item.evidence.length > 0 || Boolean(item.explanation.trim());
  });
}

function closestAmong(
  children: CriterionNode[],
  byId: Map<string, CriterionEvaluation>,
): CriterionNode | null {
  if (!children.length) return null;

  let best = children[0];
  let bestScore: [number, number, number, number] | null = null;

  children.forEach((child, index) => {
    const { met, unclear, missing } = leafStatusCounts(child, byId);
    const score: [number, number, number, number] = [
      -met,
      -unclear,
      missing,
      index,
    ];
    if (!bestScore || compareScore(score, bestScore) < 0) {
      bestScore = score;
      best = child;
    }
  });

  return best;
}

function leafStatusCounts(
  node: CriterionNode,
  byId: Map<string, CriterionEvaluation>,
): { met: number; unclear: number; missing: number } {
  let met = 0;
  let unclear = 0;
  let missing = 0;
  for (const id of collectConditionIds(node)) {
    const status = byId.get(id)?.status;
    if (status === "MET") met += 1;
    else if (status === "UNCLEAR") unclear += 1;
    else missing += 1;
  }
  return { met, unclear, missing };
}

function compareScore(
  left: [number, number, number, number],
  right: [number, number, number, number],
): number {
  for (let i = 0; i < left.length; i += 1) {
    if (left[i] !== right[i]) return left[i] - right[i];
  }
  return 0;
}

type ElevatedGap = ActionableGap & { elevatedLeafIds?: string[] };

/**
 * When a missing leaf sits under an OR parent with alternative child categories
 * entirely inside the relevant pathway, present the parent requirement.
 */
function elevateMissingToParents(
  missingItems: AnalysisResponse["missing_items"],
  root: CriterionNode | null,
  criteria: CriterionEvaluation[],
  relevantIds: Set<string>,
): ElevatedGap[] {
  if (!root || missingItems.length === 0) {
    return missingItems.map((item) => ({ ...item }));
  }

  const statusById = new Map(
    criteria.map((item) => [item.criterion_id, item.status]),
  );
  const parentById = indexParents(root);
  const usedLeaves = new Set<string>();
  const elevated: ElevatedGap[] = [];

  for (const item of missingItems) {
    if (usedLeaves.has(item.criterion_id)) continue;

    const parent = nearestRelevantOrParent(
      item.criterion_id,
      parentById,
      relevantIds,
    );
    if (!parent || !parent.description?.trim()) {
      elevated.push({ ...item });
      usedLeaves.add(item.criterion_id);
      continue;
    }

    const leafIds = collectConditionIds(parent);
    const whollyRelevant = leafIds.every((id) => relevantIds.has(id));
    const allUnmet = leafIds.every((id) => {
      const status = statusById.get(id);
      return status === "MISSING" || status === undefined;
    });
    const coversMissing = leafIds.some((id) =>
      missingItems.some((missing) => missing.criterion_id === id),
    );

    if (!whollyRelevant || !allUnmet || !coversMissing || leafIds.length < 2) {
      elevated.push({ ...item });
      usedLeaves.add(item.criterion_id);
      continue;
    }

    const childLabels = leafIds
      .map((id) => {
        const criterion = criteria.find((row) => row.criterion_id === id);
        return shortLabel(criterion?.description || id);
      })
      .filter(Boolean);

    elevated.push({
      criterion_id: parent.id,
      description: formatParentMissing(parent.description),
      detail:
        childLabels.length > 0
          ? `No ${joinLabels(childLabels)} are documented.`
          : undefined,
      elevatedLeafIds: leafIds,
    });
    for (const id of leafIds) usedLeaves.add(id);
  }

  return elevated;
}

function indexParents(root: CriterionNode): Map<string, CriterionNode> {
  const parentById = new Map<string, CriterionNode>();

  function walk(node: CriterionNode) {
    for (const child of node.children) {
      parentById.set(child.id, node);
      walk(child);
    }
  }

  walk(root);
  return parentById;
}

function nearestRelevantOrParent(
  nodeId: string,
  parentById: Map<string, CriterionNode>,
  relevantIds: Set<string>,
): CriterionNode | null {
  let current = parentById.get(nodeId) ?? null;
  while (current) {
    if (current.node_type === "OR" && current.description?.trim()) {
      const leafIds = collectConditionIds(current);
      if (leafIds.length >= 2 && leafIds.every((id) => relevantIds.has(id))) {
        return current;
      }
    }
    current = parentById.get(current.id) ?? null;
  }
  return null;
}

function collectConditionIds(node: CriterionNode): string[] {
  if (node.node_type === "CONDITION") return [node.id];
  return node.children.flatMap((child) => collectConditionIds(child));
}

function formatParentMissing(description: string): string {
  const base = description.trim().replace(/\.$/, "");
  if (/not (documented|established)\.?$/i.test(base)) {
    return base.endsWith(".") ? base : `${base}.`;
  }
  const verb = /\bsymptoms\b/i.test(base) ? "are" : "is";
  return `${base} ${verb} not established.`;
}

function shortLabel(text: string): string {
  return text
    .trim()
    .replace(/\bis not documented\.?$/i, "")
    .replace(/\.$/, "")
    .trim();
}

function joinLabels(labels: string[]): string {
  if (labels.length === 1) return labels[0];
  if (labels.length === 2) return `${labels[0]} or ${labels[1]}`;
  return `${labels.slice(0, -1).join(", ")}, or ${labels[labels.length - 1]}`;
}

/** Concise actionable next step from existing evaluations only. */
export function composeNextStep(
  result: AnalysisResponse,
  curated: {
    primaryCriteria: PrimaryCriterion[];
    missingItems: ActionableGap[];
    reviewItems: AnalysisResponse["review_items"];
    triggered: AnalysisResponse["exclusions"];
  },
): string {
  if (
    result.overall_status === "NOT_APPLICABLE" ||
    result.overall_status === "READY"
  ) {
    return result.next_step;
  }

  const gap = selectGap(
    curated.missingItems,
    curated.primaryCriteria,
    curated.reviewItems,
  );
  const gapLabel = gap ? shortRequirement(gap.label) : null;

  if (curated.triggered.length > 0) {
    if (gap && gapLabel) {
      const reviewFocus = gap.absenceDocumented
        ? `Review the documented absence of ${gapLabel} with the treating clinician.`
        : gap.hasEvidence
          ? `Review the documented findings related to ${gapLabel} with the treating clinician.`
          : `Review ${gapLabel} with the treating clinician.`;
      return [
        reviewFocus,
        `If supporting evidence for this requirement is documented elsewhere in the chart, add that evidence before submission.`,
        `If it remains unmet, the policy non-indication remains blocking.`,
      ].join(" ");
    }
    const exclusionFocus = shortRequirement(
      curated.triggered[0]?.explanation || "the triggered policy non-indication",
    );
    return [
      `Review the triggered policy non-indication (${exclusionFocus}) with the treating clinician.`,
      `If chart evidence that resolves this non-indication is documented elsewhere, add it before submission.`,
      `Otherwise the non-indication remains blocking.`,
    ].join(" ");
  }

  if (gap && gapLabel) {
    if (result.overall_status === "NEEDS_REVIEW" || gap.kind === "unclear") {
      return [
        `Manual review is required for ${gapLabel}.`,
        `Confirm whether supporting chart evidence exists and include it before submission if available.`,
      ].join(" ");
    }
    return [
      `Obtain or locate documentation for ${gapLabel} before submission.`,
      `If supporting evidence is already in the chart, add it to the packet; if it remains absent, readiness stays blocked.`,
    ].join(" ");
  }

  return result.next_step;
}

function selectGap(
  missingItems: ActionableGap[],
  primaryCriteria: PrimaryCriterion[],
  reviewItems: AnalysisResponse["review_items"],
): {
  label: string;
  absenceDocumented: boolean;
  hasEvidence: boolean;
  kind: "missing" | "unclear";
} | null {
  if (missingItems.length > 0) {
    const missing = missingItems[0];
    const criterion = primaryCriteria.find(
      (item) => item.criterion_id === missing.criterion_id,
    );
    const label =
      shortRequirement(missing.description) ||
      criterion?.description?.trim() ||
      missing.description;
    const absenceDocumented =
      criterion?.display_status === "NOT_SATISFIED" ||
      hasAffirmativeConflict(
        `${criterion?.explanation ?? ""} ${missing.description}`,
      );
    return {
      label,
      absenceDocumented,
      hasEvidence: (criterion?.evidence.length ?? 0) > 0,
      kind: "missing",
    };
  }

  const unclearPrimary = primaryCriteria.find(
    (item) => item.status === "UNCLEAR" || item.display_status === "UNCLEAR",
  );
  if (unclearPrimary) {
    return {
      label:
        unclearPrimary.description?.trim() ||
        shortRequirement(unclearPrimary.explanation) ||
        "the flagged requirement",
      absenceDocumented: false,
      hasEvidence: unclearPrimary.evidence.length > 0,
      kind: "unclear",
    };
  }

  if (reviewItems.length > 0) {
    return {
      label:
        shortRequirement(reviewItems[0].description) || "the flagged requirement",
      absenceDocumented: false,
      hasEvidence: false,
      kind: "unclear",
    };
  }

  return null;
}

function shortRequirement(text: string): string {
  const cleaned = text
    .replace(/\bis not documented\.?$/i, "")
    .replace(/\bare not established\.?$/i, "")
    .replace(/\bis not established\.?$/i, "")
    .replace(
      /\bis affirmatively not met by the available chart evidence\.?$/i,
      "",
    )
    .replace(/^policy non-indication triggered:?\s*/i, "")
    .trim()
    .replace(/\.$/, "");
  if (!cleaned) return text.trim().replace(/\.$/, "");
  if (cleaned.length <= 120) return cleaned;
  return `${cleaned.slice(0, 117).trim()}...`;
}

function hasAffirmativeConflict(text: string): boolean {
  const lowered = text.toLowerCase();
  return AFFIRMATIVE_TOKENS.some((token) => lowered.includes(token));
}

function deprioritizeGenericMet(
  criteria: CriterionEvaluation[],
  options: { hasActionableGap: boolean },
): CriterionEvaluation[] {
  if (!options.hasActionableGap) return criteria;
  const specific = criteria.filter(
    (item) => !(item.status === "MET" && isGenericDocumentation(item)),
  );
  return specific.length > 0 ? specific : criteria;
}

function isGenericDocumentation(item: CriterionEvaluation): boolean {
  const text = `${item.description ?? ""} ${item.explanation}`.toLowerCase();
  return (
    /\bis provided\b/.test(text) ||
    /\bclinically relevant\b/.test(text) ||
    /\bcurrent clinical information\b/.test(text)
  );
}

function displayStatus(
  item: CriterionEvaluation,
  missing?: ActionableGap,
): DisplayStatus {
  if (item.status === "MET") return "MET";
  if (item.status === "UNCLEAR") return "UNCLEAR";
  const haystack =
    `${item.explanation} ${missing?.description ?? ""}`.toLowerCase();
  if (AFFIRMATIVE_TOKENS.some((token) => haystack.includes(token))) {
    return "NOT_SATISFIED";
  }
  return "MISSING";
}

function primarySortKey(item: PrimaryCriterion): number {
  switch (item.display_status) {
    case "MISSING":
    case "NOT_SATISFIED":
      return 0;
    case "UNCLEAR":
      return 1;
    case "MET":
      return 2;
    default:
      return 3;
  }
}
