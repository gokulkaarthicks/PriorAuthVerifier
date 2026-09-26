import { expect, test, type Page } from "@playwright/test";
import type { AnalysisResponse } from "../src/types/analysis";
const pdf = {
  name: "chart.pdf",
  mimeType: "application/pdf",
  buffer: Buffer.from("%PDF-1.7 test"),
};
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
const result: AnalysisResponse = {
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
async function upload(page: Page) {
  await page.locator('input[aria-label="Patient Chart"]').setInputFiles(pdf);
  await page
    .locator('input[aria-label="Policy / Guideline"]')
    .setInputFiles({ ...pdf, name: "policy.pdf" });
}
test("file selection, removal, PDF rejection, and explicit submission", async ({
  page,
}) => {
  let requests = 0;
  await page.route("**/api/analyze", (route) => {
    requests++;
    return route.fulfill({ json: result });
  });
  await page.goto("/");
  const run = page.getByRole("button", { name: "Run verification" });
  await expect(run).toBeDisabled();
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose patient chart" }).click();
  await (await chooser).setFiles(pdf);
  await expect(run).toBeDisabled();
  await page.locator('input[aria-label="Policy / Guideline"]').setInputFiles({
    name: "bad.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("invalid"),
  });
  await expect(page.getByRole("alert")).toContainText("Choose a PDF");
  await upload(page);
  await expect(run).toBeEnabled();
  expect(requests).toBe(0);
  await page
    .getByRole("button", { name: "Remove Patient Chart", exact: true })
    .click();
  await expect(run).toBeDisabled();
  await upload(page);
  await run.click();
  await expect(
    page.getByRole("heading", { name: "Verification results" }),
  ).toBeVisible();
  expect(requests).toBe(1);
  await expect(
    page.getByRole("heading", { name: "Blocking policy findings" }),
  ).toBeVisible();
  await expect(
    page.getByText("Patient Chart · Page 4 · Nov 12, 2025"),
  ).toBeVisible();
  await expect(
    page.getByText("Non-triggered policy check", { exact: true }),
  ).not.toBeVisible();
  await page.getByText("View all policy checks").click();
  await expect(
    page.getByText("Non-triggered policy check", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("body")).not.toContainText("internal-");
  expect(
    await page.evaluate(() => [localStorage.length, sessionStorage.length]),
  ).toEqual([0, 0]);
  await page.screenshot({
    path: "test-results/review-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "New verification" }).click();
  await expect(run).toBeDisabled();
});
test("long analysis retains filenames, blocks repeats, shows real elapsed time", async ({
  page,
}) => {
  let finish: () => void = () => {};
  const pending = new Promise<void>((resolve) => {
    finish = resolve;
  });
  await page.route("**/api/analyze", async (route) => {
    await pending;
    await route.fulfill({ json: result });
  });
  await page.clock.install();
  await page.goto("/");
  await upload(page);
  await page.getByRole("button", { name: "Run verification" }).click();
  await expect(page.getByRole("progressbar")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Verification in progress" }),
  ).toBeDisabled();
  await expect(page.getByText("chart.pdf", { exact: true })).toBeVisible();
  await page.clock.fastForward(65000);
  await expect(page.getByText("01:05 elapsed")).toBeVisible();
  await expect(
    page.getByText(
      "Large policy documents can take several minutes to evaluate.",
    ),
  ).toBeVisible();
  finish();
  await expect(
    page.getByRole("heading", { name: "Verification results" }),
  ).toBeVisible();
});
for (const status of ["READY", "NOT_APPLICABLE", "NEEDS_REVIEW"] as const) {
  test(`faithfully renders ${status} without computing readiness`, async ({
    page,
  }) => {
    await page.route("**/api/analyze", (route) =>
      route.fulfill({
        json: {
          ...result,
          overall_status: status,
          matching_policy_criteria_found: status !== "NOT_APPLICABLE",
          exclusions: [],
          criteria: [],
          missing_items: [],
        },
      }),
    );
    await page.goto("/");
    await upload(page);
    await page.getByRole("button", { name: "Run verification" }).click();
    await expect(page.locator(".outcome .status-badge")).toHaveText(
      status.replaceAll("_", " "),
    );
    await expect(page.getByText("33208", { exact: true })).toBeVisible();
    if (status === "NOT_APPLICABLE")
      await expect(
        page.getByRole("heading", { name: "No matching procedure criteria" }),
      ).toBeVisible();
    if (status === "READY")
      await expect(
        page.getByText(
          "Documentation satisfies the evaluated requirements of the uploaded policy.",
        ),
      ).toBeVisible();
  });
}
for (const status of [400, 422, 413, 500, 502, 503]) {
  test(`safe error for HTTP ${status}`, async ({ page }) => {
    let requests = 0;
    await page.route("**/api/analyze", (route) => {
      requests++;
      return route.fulfill({
        status,
        json: {
          error: { message: "SECRET /private/server.py raw provider error" },
        },
      });
    });
    await page.goto("/");
    await upload(page);
    await page.getByRole("button", { name: "Run verification" }).click();
    await expect(page.getByRole("alert")).toContainText(
      "Unable to complete verification",
    );
    await expect(page.locator("body")).not.toContainText("SECRET");
    await expect(
      page.getByRole("button", { name: "Run verification" }),
    ).toBeEnabled();
    expect(requests).toBe(1);
  });
}
test("network and malformed responses fail safely", async ({ page }) => {
  await page.route("**/api/analyze", (route) => route.abort());
  await page.goto("/");
  await upload(page);
  await page.getByRole("button", { name: "Run verification" }).click();
  await expect(page.getByRole("alert")).toContainText("Connection lost");
  await page.unroute("**/api/analyze");
  await page.route("**/api/analyze", (route) =>
    route.fulfill({ json: { overall_status: "APPROVED" } }),
  );
  await page.getByRole("button", { name: "Run verification" }).click();
  await expect(page.getByRole("alert")).toContainText("unreadable result");
});
test("mobile layout fits and disclosures work with keyboard", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.route("**/api/analyze", (route) =>
    route.fulfill({ json: result }),
  );
  await page.goto("/");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await upload(page);
  await page.getByRole("button", { name: "Run verification" }).click();
  await expect(
    page.getByRole("heading", { name: "Verification results" }),
  ).toBeVisible();
  const disclosure = page
    .locator("summary")
    .filter({ hasText: "All policy criteria" });
  await disclosure.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText("CONDITION", { exact: true })).toBeVisible();
  await page.screenshot({
    path: "test-results/review-mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
