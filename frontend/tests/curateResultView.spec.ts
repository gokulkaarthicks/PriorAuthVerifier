import { expect, test } from "@playwright/test";
import type {
  AnalysisResponse,
  CriterionEvaluation,
  CriterionNode,
} from "../src/types/analysis";
import { curateResultView } from "../src/utils/curateResultView";
import { result } from "./fixtures";

test("unsupported OR branches are excluded from primary curation", () => {
  const curated = curateResultView(result);

  const primaryText = curated.primaryCriteria
    .map((item) => item.description || "")
    .join("\n");
  const missingText = curated.missingItems
    .map((item) => `${item.description}\n${item.detail || ""}`)
    .join("\n");

  expect(primaryText).toContain("Heart rate below 50 beats per minute");
  expect(primaryText).toContain("Sinus pause greater than 3 seconds");
  expect(primaryText).toContain(
    "Qualifying symptoms attributable to sinus node dysfunction",
  );
  expect(primaryText).not.toContain(
    "Associated conduction disorder, with or without symptoms",
  );
  expect(primaryText).not.toContain(
    "Progressive neuromuscular disease known to involve the heart",
  );
  expect(primaryText).not.toContain("Mobitz II AV block");

  expect(missingText).toContain(
    "Qualifying symptoms attributable to sinus node dysfunction are not established.",
  );
  expect(missingText).not.toContain(
    "Associated conduction disorder, with or without symptoms",
  );
  expect(missingText).not.toContain(
    "Progressive neuromuscular disease known to involve the heart",
  );
  expect(
    curated.missingItems.every((item) =>
      ["symptoms-parent", "syncope", "hf", "fatigue"].includes(item.criterion_id),
    ),
  ).toBe(true);

  // Full criteria payload still retains unsupported branches for audit view.
  expect(
    result.criteria.some((item) => item.criterion_id === "conduction"),
  ).toBe(true);
  expect(result.criteria.some((item) => item.criterion_id === "neuro")).toBe(
    true,
  );
});

test("full-tree MISSING outside curated pathway is omitted from primary missing panel", () => {
  const emptyEval = {
    temporal_warnings: [] as string[],
    evidence_chunk_ids: [] as string[],
    evidence_text: [] as string[],
    page_numbers: [] as number[],
  };

  const met = (
    criterion_id: string,
    description: string,
  ): CriterionEvaluation => ({
    criterion_id,
    description,
    status: "MET",
    explanation: "Documented in chart.",
    evidence: [
      {
        chunk_id: `chunk-${criterion_id}`,
        text: description,
        page_number: 1,
        document_date: null,
      },
    ],
    ...emptyEval,
  });

  const missing = (
    criterion_id: string,
    description: string,
  ): CriterionEvaluation => ({
    criterion_id,
    description,
    status: "MISSING",
    explanation: "No qualifying documentation found.",
    evidence: [],
    ...emptyEval,
  });

  // Root AND mirrors policies that attach an unsupported sibling group beside
  // the selected indication pathway. Backend missing_items may still list that
  // sibling leaf; primary presentation must not.
  const criteriaTree: CriterionNode = {
    id: "root",
    node_type: "AND",
    description: "Documentation and an applicable indication",
    children: [
      {
        id: "docs",
        node_type: "CONDITION",
        description: "Current clinical information is provided.",
        children: [],
      },
      {
        id: "indications",
        node_type: "OR",
        description: "Applicable indication",
        children: [
          {
            id: "pathway-a",
            node_type: "AND",
            description: "Supported indication pathway",
            children: [
              {
                id: "finding",
                node_type: "CONDITION",
                description: "Primary clinical finding",
                children: [],
              },
              {
                id: "gap-parent",
                node_type: "OR",
                description: "Attributable symptoms",
                children: [
                  {
                    id: "gap-leaf",
                    node_type: "CONDITION",
                    description: "Attributable syncope",
                    children: [],
                  },
                  {
                    id: "gap-alt",
                    node_type: "CONDITION",
                    description: "Attributable fatigue",
                    children: [],
                  },
                ],
              },
            ],
          },
          {
            id: "pathway-b",
            node_type: "AND",
            description: "Alternative indication pathway",
            children: [
              {
                id: "alt-finding",
                node_type: "CONDITION",
                description: "Unrelated alternative finding",
                children: [],
              },
            ],
          },
        ],
      },
      {
        id: "sibling-group",
        node_type: "AND",
        description: "Unsupported sibling circumstance",
        children: [
          {
            id: "off-pathway-missing",
            node_type: "CONDITION",
            description: "Off-pathway requirement from full criteria tree",
            children: [],
          },
        ],
      },
    ],
  };

  const synthetic: AnalysisResponse = {
    ...result,
    criteria: [
      met("docs", "Current clinical information is provided."),
      met("finding", "Primary clinical finding"),
      missing("gap-leaf", "Attributable syncope"),
      missing("gap-alt", "Attributable fatigue"),
      missing("alt-finding", "Unrelated alternative finding"),
      missing(
        "off-pathway-missing",
        "Off-pathway requirement from full criteria tree",
      ),
    ],
    exclusions: [],
    missing_items: [
      {
        criterion_id: "gap-leaf",
        description: "Attributable syncope is not documented.",
      },
      {
        criterion_id: "off-pathway-missing",
        description:
          "Off-pathway requirement from full criteria tree is not documented.",
      },
    ],
    review_items: [],
    criteria_tree: criteriaTree,
    overall_status: "NOT_READY",
    next_step: "Obtain documentation for the missing requirement.",
  };

  const curated = curateResultView(synthetic);
  const missingText = curated.missingItems
    .map(
      (item) =>
        `${item.criterion_id}|${item.description}|${item.detail || ""}`,
    )
    .join("\n");

  expect(missingText).toContain("Attributable symptoms are not established.");
  expect(missingText).not.toContain("off-pathway-missing");
  expect(missingText).not.toContain(
    "Off-pathway requirement from full criteria tree",
  );
  expect(
    curated.missingItems.every((item) =>
      ["gap-parent", "gap-leaf", "gap-alt"].includes(item.criterion_id),
    ),
  ).toBe(true);
  expect(
    synthetic.missing_items.some(
      (item) => item.criterion_id === "off-pathway-missing",
    ),
  ).toBe(true);
  expect(
    synthetic.criteria.some(
      (item) => item.criterion_id === "off-pathway-missing",
    ),
  ).toBe(true);
});

test("NOT_APPLICABLE produces no primary pathway curation", () => {
  const curated = curateResultView({
    ...result,
    matching_policy_criteria_found: false,
    overall_status: "NOT_APPLICABLE",
    criteria: [],
    exclusions: [],
    missing_items: [],
    review_items: [],
    criteria_tree: null,
    next_step:
      "No criteria for the detected planned procedure were found in the uploaded policy.",
  });
  expect(curated.primaryCriteria).toEqual([]);
  expect(curated.missingItems).toEqual([]);
  expect(curated.sourceDateNotes).toEqual([]);
});
