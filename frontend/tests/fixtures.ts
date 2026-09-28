import type { AnalysisResponse, CriterionNode } from "../src/types/analysis";

const emptyEval = {
  temporal_warnings: [] as string[],
  evidence_chunk_ids: [] as string[],
  evidence_text: [] as string[],
  page_numbers: [] as number[],
};

const hrCriterion = {
  criterion_id: "hr",
  description: "Heart rate below 50 beats per minute",
  status: "MET" as const,
  explanation: "Mean HR 46 bpm / ECG HR 44 bpm",
  evidence: [
    {
      chunk_id: "chunk-hr",
      text: "Heart rate 44 bpm",
      page_number: 5,
      document_date: "2025-10-08",
    },
  ],
  ...emptyEval,
  temporal_warnings: [
    "Conflicting source dates were found for the Holter record.",
  ],
};

const pauseCriterion = {
  criterion_id: "pause",
  description: "Sinus pause greater than 3 seconds",
  status: "MET" as const,
  explanation: "Longest pause 3.1 seconds",
  evidence: [
    {
      chunk_id: "chunk-pause",
      text: "Longest pause 3.1 seconds",
      page_number: 5,
      document_date: "2025-10-08",
    },
  ],
  ...emptyEval,
  temporal_warnings: [
    "Encounter date differs from Holter report header date.",
  ],
};

const syncopeCriterion = {
  criterion_id: "syncope",
  description: "Attributable syncope or presyncope",
  status: "MISSING" as const,
  explanation:
    "Patient is documented as asymptomatic with no functional limitation.",
  evidence: [
    {
      chunk_id: "chunk-symptoms",
      text: "Patient remains fully asymptomatic with no functional limitation.",
      page_number: 4,
      document_date: "2025-11-12",
    },
  ],
  ...emptyEval,
};

const hfCriterion = {
  criterion_id: "hf",
  description: "Heart-failure symptoms attributable to bradycardia",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const fatigueCriterion = {
  criterion_id: "fatigue",
  description: "Exertional fatigue or impaired exercise tolerance",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const mobitzCriterion = {
  criterion_id: "mobitz",
  description: "Mobitz II AV block",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const congenitalCriterion = {
  criterion_id: "congenital",
  description: "Congenital complete or high-degree AV block",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const taviCriterion = {
  criterion_id: "tavi",
  description: "TAVI-related conduction disturbance",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const transplantCriterion = {
  criterion_id: "transplant",
  description: "Cardiac transplant conditions",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const conductionCriterion = {
  criterion_id: "conduction",
  description: "Associated conduction disorder, with or without symptoms",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const neuromuscularCriterion = {
  criterion_id: "neuro",
  description: "Progressive neuromuscular disease known to involve the heart",
  status: "MISSING" as const,
  explanation: "No qualifying documentation found.",
  evidence: [],
  ...emptyEval,
};

const docsCriterion = {
  criterion_id: "docs",
  description: "Current clinical information is provided.",
  status: "MET" as const,
  explanation: "Chart contains current clinical information.",
  evidence: [
    {
      chunk_id: "chunk-docs",
      text: "Progress note dated 2025-11-12",
      page_number: 1,
      document_date: "2025-11-12",
    },
  ],
  ...emptyEval,
};

const criteriaTree: CriterionNode = {
  id: "root",
  node_type: "OR",
  description: null,
  children: [
    {
      id: "snd",
      node_type: "AND",
      description: "Sinus node dysfunction pathway",
      children: [
        {
          id: "hr",
          node_type: "CONDITION",
          description: hrCriterion.description,
          children: [],
        },
        {
          id: "pause",
          node_type: "CONDITION",
          description: pauseCriterion.description,
          children: [],
        },
        {
          id: "symptoms-parent",
          node_type: "OR",
          description: "Qualifying symptoms attributable to sinus node dysfunction",
          children: [
            {
              id: "syncope",
              node_type: "CONDITION",
              description: syncopeCriterion.description,
              children: [],
            },
            {
              id: "hf",
              node_type: "CONDITION",
              description: hfCriterion.description,
              children: [],
            },
            {
              id: "fatigue",
              node_type: "CONDITION",
              description: fatigueCriterion.description,
              children: [],
            },
          ],
        },
      ],
    },
    {
      id: "avb",
      node_type: "AND",
      description: "Atrioventricular block pathway",
      children: [
        {
          id: "mobitz",
          node_type: "CONDITION",
          description: mobitzCriterion.description,
          children: [],
        },
        {
          id: "congenital",
          node_type: "CONDITION",
          description: congenitalCriterion.description,
          children: [],
        },
      ],
    },
    {
      id: "conduction-branch",
      node_type: "AND",
      description: "Associated conduction disorder, with or without symptoms",
      children: [
        {
          id: "conduction",
          node_type: "CONDITION",
          description: conductionCriterion.description,
          children: [],
        },
      ],
    },
    {
      id: "neuro-branch",
      node_type: "AND",
      description: "Progressive neuromuscular disease known to involve the heart",
      children: [
        {
          id: "neuro",
          node_type: "CONDITION",
          description: neuromuscularCriterion.description,
          children: [],
        },
      ],
    },
  ],
};

export const result: AnalysisResponse = {
  procedure: {
    procedure_name: "Permanent Cardiac Pacemaker Implantation",
    cpt_code: "33207",
    status: "planned",
    source_chunk_id: null,
    source_text: null,
    page_number: 4,
    is_future_procedure: true,
    confidence: 0.99,
  },
  matching_policy_criteria_found: true,
  overall_status: "NOT_READY",
  criteria: [
    docsCriterion,
    hrCriterion,
    pauseCriterion,
    syncopeCriterion,
    hfCriterion,
    fatigueCriterion,
    mobitzCriterion,
    congenitalCriterion,
    conductionCriterion,
    neuromuscularCriterion,
    taviCriterion,
    transplantCriterion,
  ],
  exclusions: [
    {
      exclusion_id: "ex-asymptomatic",
      triggered: true,
      status: "MET",
      explanation:
        "Sinus bradycardia and pauses are documented while the patient remains asymptomatic.",
      evidence: [
        {
          chunk_id: "private-id",
          text: "Patient remains fully asymptomatic with no functional limitation.",
          page_number: 4,
          document_date: "2025-11-12",
        },
      ],
      ...emptyEval,
      temporal_warnings: [
        "Conflicting source dates were found for the Holter record.",
      ],
    },
    {
      exclusion_id: "ex-other",
      triggered: false,
      status: "MISSING",
      explanation: "Non-triggered neuromuscular disease check",
      evidence: [],
      ...emptyEval,
    },
  ],
  // Includes unsupported-branch leaves to prove primary curation drops them.
  missing_items: [
    {
      criterion_id: "syncope",
      description: "Attributable syncope or presyncope is not documented.",
    },
    {
      criterion_id: "conduction",
      description:
        "Associated conduction disorder, with or without symptoms is not documented.",
    },
    {
      criterion_id: "neuro",
      description:
        "Progressive neuromuscular disease known to involve the heart is not documented.",
    },
  ],
  review_items: [
    {
      criterion_id: "ex-asymptomatic",
      description: "Policy non-indication triggered",
      explanation:
        "Sinus bradycardia and pauses are documented while the patient remains asymptomatic.",
    },
  ],
  next_step:
    "Manual review is required because a policy non-indication is documented.",
  criteria_tree: criteriaTree,
};

export const notApplicableResult: AnalysisResponse = {
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
};
