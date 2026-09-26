import { expect, test, type Page } from "@playwright/test";
import type { JobDetail, JobSummary } from "../src/types/jobs";
import { result } from "./fixtures";

const pdf = {
  name: "chart.pdf",
  mimeType: "application/pdf",
  buffer: Buffer.from("%PDF-1.7 test"),
};
function job(id: string, status: JobSummary["status"] = "QUEUED"): JobDetail {
  return {
    job_id: id,
    status,
    created_at: new Date().toISOString(),
    started_at: status === "RUNNING" ? new Date().toISOString() : null,
    completed_at:
      status === "COMPLETED" || status === "FAILED"
        ? new Date().toISOString()
        : null,
    patient_filename: `${id}.pdf`,
    policy_filename: "policy.pdf",
    procedure_name:
      status === "COMPLETED" ? result.procedure.procedure_name : null,
    cpt_code: status === "COMPLETED" ? "33208" : null,
    overall_status: status === "COMPLETED" ? result.overall_status : null,
    error_message: status === "FAILED" ? "SECRET provider traceback" : null,
    result: status === "COMPLETED" ? result : null,
  };
}
async function mockQueue(page: Page, jobs: JobDetail[] = []) {
  const state = {
    jobs,
    posts: 0,
    gets: 0,
    details: 0,
    postStatus: 202,
    listFailure: false,
    postAbort: false,
  };
  await page.route(/\/api\/jobs(?:\/[^/?]+)?(?:\?.*)?$/, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/api/jobs" && request.method() === "POST") {
      state.posts++;
      expect(request.postData()).toContain('name="patient_file"');
      expect(request.postData()).toContain('name="policy_file"');
      if (state.postAbort) return route.abort();
      if (state.postStatus !== 202)
        return route.fulfill({
          status: state.postStatus,
          json: { detail: "SECRET raw provider error /private/file" },
        });
      const next = job(`case-${state.posts}`);
      state.jobs.unshift(next);
      const { result: _, ...metadata } = next;
      return route.fulfill({ status: 202, json: metadata });
    }
    if (path === "/api/jobs") {
      state.gets++;
      if (state.listFailure) return route.fulfill({ status: 503 });
      return route.fulfill({
        json: {
          jobs: state.jobs.map(({ result: _, ...metadata }) => metadata),
          active_count: state.jobs.filter(
            (j) => j.status === "RUNNING" || j.status === "QUEUED",
          ).length,
          max_active_jobs: 20,
          execution_concurrency: 2,
        },
      });
    }
    state.details++;
    const found = state.jobs.find((j) => path === `/api/jobs/${j.job_id}`);
    return found
      ? route.fulfill({ json: found })
      : route.fulfill({ status: 404 });
  });
  return state;
}
async function upload(page: Page) {
  await page.locator('input[aria-label="Patient chart"]').setInputFiles(pdf);
  await page
    .locator('input[aria-label="Policy / guideline"]')
    .setInputFiles({ ...pdf, name: "policy.pdf" });
}
const startButton = (page: Page) =>
  page.getByRole("button", { name: "Start verification", exact: true });

test("PDF controls validate files, remove files, and submit only on click", async ({
  page,
}) => {
  const state = await mockQueue(page);
  await page.goto("/");
  await expect(startButton(page)).toBeDisabled();
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose patient chart" }).click();
  await (await chooser).setFiles(pdf);
  await expect(startButton(page)).toBeDisabled();
  await page.locator('input[aria-label="Policy / guideline"]').setInputFiles({
    name: "bad.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("invalid"),
  });
  await expect(page.getByRole("alert")).toContainText("Choose a PDF");
  await upload(page);
  await expect(startButton(page)).toBeEnabled();
  expect(state.posts).toBe(0);
  await page
    .getByRole("button", { name: "Remove Patient chart", exact: true })
    .click();
  await expect(startButton(page)).toBeDisabled();
  await upload(page);
  await startButton(page).click();
  await expect(
    page.getByRole("cell", { name: "case-1.pdf", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Choose patient chart" }),
  ).toBeEnabled();
  await expect(startButton(page)).toBeDisabled();
  expect(state.posts).toBe(1);
  expect(
    await page.evaluate(() => [localStorage.length, sessionStorage.length]),
  ).toEqual([0, 0]);
});

test("multiple submissions stay listed while queued and analyzing; form stays available", async ({
  page,
}) => {
  await page.clock.install();
  const state = await mockQueue(page);
  await page.goto("/");
  await upload(page);
  await startButton(page).click();
  await expect(
    page.getByRole("cell", { name: "case-1.pdf", exact: true }),
  ).toBeVisible();
  state.jobs[0] = job("case-1", "RUNNING");
  await page.clock.fastForward(3000);
  await expect(page.getByText(/Analyzing/)).toBeVisible();
  await upload(page);
  await startButton(page).click();
  await expect(
    page.getByRole("cell", { name: "case-2.pdf", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Queued", { exact: true })).toBeVisible();
  await expect(page.getByText("2 active / 20", { exact: true })).toBeVisible();
  await page.clock.fastForward(65000);
  await expect(page.getByText(/Analyzing/)).toContainText("01:");
  await expect(
    page.getByRole("heading", { name: "New prior authorization", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Choose patient chart" }),
  ).toBeEnabled();
  expect(state.posts).toBe(2);
  await page.getByRole("button", { name: /Recent cases/ }).click();
  await expect(
    page.getByRole("cell", { name: "case-1.pdf", exact: true }),
  ).not.toBeVisible();
  await page.getByRole("button", { name: /Recent cases/ }).click();
  await expect(
    page.getByRole("cell", { name: "case-1.pdf", exact: true }),
  ).toBeVisible();
});

test("review/back/new/reload retain server history and open the correct result", async ({
  page,
}) => {
  const a = job("alice", "COMPLETED");
  const b = job("bob", "COMPLETED");
  b.result = {
    ...result,
    procedure: {
      ...result.procedure,
      procedure_name: "Other procedure",
      cpt_code: "12345",
    },
  };
  await mockQueue(page, [b, a, job("running", "RUNNING")]);
  await page.goto("/");
  await upload(page);
  await page
    .getByRole("button", { name: "Review alice.pdf", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Prior authorization review",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByText("33208", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Blocking policy findings" }),
  ).toBeVisible();
  await expect(
    page.getByText("Patient Chart · Page 4 · Nov 12, 2025"),
  ).toBeVisible();
  const quote = page.getByText(
    "“Patient remains fully asymptomatic with no functional limitation.”",
    { exact: true },
  );
  await expect(quote).not.toBeVisible();
  await page.locator(".blocking summary").click();
  await expect(quote).toBeVisible();
  await expect(page.locator("body")).not.toContainText("internal-");
  await page
    .getByRole("button", { name: "← Back to prior authorization cases" })
    .click();
  await expect(
    page.getByRole("button", { name: "Review alice.pdf" }),
  ).toBeVisible();
  await expect(startButton(page)).toBeEnabled(); // Upload draft also survives review.
  await page.getByRole("button", { name: "Review bob.pdf" }).click();
  await expect(page.getByText("12345", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "New prior authorization" }).click();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Review alice.pdf" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Review alice.pdf" }).click();
  await expect(page.getByText("33208", { exact: true })).toBeVisible();
});

test("capacity returns after completion and polling stops once idle", async ({
  page,
}) => {
  await page.clock.install();
  const state = await mockQueue(
    page,
    Array.from({ length: 20 }, (_, i) =>
      job(`case-${i}`, i < 2 ? "RUNNING" : "QUEUED"),
    ),
  );
  await page.goto("/");
  await upload(page);
  await expect(startButton(page)).toBeDisabled();
  await expect(page.getByText("20 active / 20", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Wait for a case to finish before starting another."),
  ).toBeVisible();
  state.jobs[0] = job("case-0", "COMPLETED");
  await page.clock.fastForward(3000);
  await expect(page.getByText("19 active / 20", { exact: true })).toBeVisible();
  await expect(startButton(page)).toBeEnabled();
  state.jobs = state.jobs.map((j) => job(j.job_id, "COMPLETED"));
  await page.clock.fastForward(3000);
  await expect(page.getByText("0 active / 20", { exact: true })).toBeVisible();
  const count = state.gets;
  await page.clock.fastForward(15000);
  expect(state.gets).toBe(count);
});

test("failed cases remain visible with safe details", async ({ page }) => {
  await mockQueue(page, [job("failed", "FAILED")]);
  await page.goto("/");
  await expect(page.getByText("Failed", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Details" }).click();
  await expect(
    page.getByText("Analysis could not be completed.", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("body")).not.toContainText("SECRET");
  await expect(page.getByText("0 active / 20", { exact: true })).toBeVisible();
});

for (const status of [
  "READY",
  "NOT_READY",
  "NOT_APPLICABLE",
  "NEEDS_REVIEW",
] as const) {
  test(`completed ${status} uses backend status faithfully`, async ({
    page,
  }) => {
    const completed = job("completed", "COMPLETED");
    completed.overall_status = status;
    completed.result = {
      ...result,
      overall_status: status,
      matching_policy_criteria_found: status !== "NOT_APPLICABLE",
      exclusions: [],
      criteria: [],
      missing_items: [],
    };
    await mockQueue(page, [completed]);
    await page.goto("/");
    await expect(page.locator(".case-table .status-badge")).toHaveText(
      status.replaceAll("_", " "),
    );
    await page.getByRole("button", { name: "Review completed.pdf" }).click();
    await expect(page.locator(".outcome .status-badge")).toHaveText(
      status.replaceAll("_", " "),
    );
    if (status === "NOT_APPLICABLE")
      await expect(
        page.getByRole("heading", { name: "No matching procedure criteria" }),
      ).toBeVisible();
  });
}
for (const code of [400, 422, 413, 429, 500, 502, 503]) {
  test(`HTTP ${code} submission error preserves selected files without retry`, async ({
    page,
  }) => {
    const state = await mockQueue(page);
    state.postStatus = code;
    await page.goto("/");
    await upload(page);
    await startButton(page).click();
    await expect(page.getByRole("alert")).toBeVisible();
    await expect(page.locator("body")).not.toContainText("SECRET");
    await expect(page.getByText("chart.pdf", { exact: true })).toBeVisible();
    expect(state.posts).toBe(1);
  });
}

test("polling failures recover; uncertain submission is never retried", async ({
  page,
}) => {
  await page.clock.install();
  const state = await mockQueue(page);
  state.listFailure = true;
  await page.goto("/");
  await expect(
    page.getByText("Case list unavailable. Reconnecting…"),
  ).toBeVisible();
  state.listFailure = false;
  await page.clock.fastForward(6000);
  await expect(page.getByText("0 active / 20", { exact: true })).toBeVisible();
  state.postAbort = true;
  await upload(page);
  await startButton(page).click();
  await expect(page.getByRole("alert")).toContainText("Connection lost");
  await page.clock.fastForward(10000);
  expect(state.posts).toBe(1);
});

test("mobile queue and review fit; source disclosures are keyboard accessible", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await mockQueue(page, [
    job("ready", "COMPLETED"),
    job("queued"),
    job("running", "RUNNING"),
  ]);
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Review ready.pdf" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/queue-mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Review ready.pdf" }).click();
  const disclosure = page.locator(".blocking summary");
  await disclosure.focus();
  await page.keyboard.press("Enter");
  await expect(
    page.getByText(
      "“Patient remains fully asymptomatic with no functional limitation.”",
    ),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/review-mobile.png",
    fullPage: true,
  });
});

test("desktop queue stays compact and source evidence is collapsed", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await mockQueue(page, [
    job("queued"),
    job("running", "RUNNING"),
    job("not-ready", "COMPLETED"),
    job("failed", "FAILED"),
  ]);
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Review not-ready.pdf" }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/queue-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Review not-ready.pdf" }).click();
  await expect(
    page.getByRole("heading", { name: "Prior authorization review" }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/review-desktop.png",
    fullPage: true,
  });
});
