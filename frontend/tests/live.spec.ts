import { expect, test } from "@playwright/test";

// Explicitly opt in: calls the configured providers once with the sample PDFs.
test("live UHC sample through the upload workflow", async ({ page }) => {
  test.skip(
    process.env.RUN_LIVE_UHC !== "1",
    "Paid, large-policy smoke test is opt-in",
  );
  test.setTimeout(15 * 60 * 1000);
  await page.goto("/");
  await page
    .locator('input[aria-label="Patient Chart"]')
    .setInputFiles("../sample_data/patient.pdf");
  await page
    .locator('input[aria-label="Policy / Guideline"]')
    .setInputFiles("../sample_data/uhc_policy.pdf");
  const responsePromise = page.waitForResponse(
    (response) => response.url().endsWith("/api/analyze"),
    { timeout: 14 * 60 * 1000 },
  );
  await page.getByRole("button", { name: "Run verification" }).click();
  await expect(
    page.getByRole("heading", { name: "Analyzing authorization" }),
  ).toBeVisible();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const result = await response.json();
  await expect(
    page.getByRole("heading", { name: "Verification results" }),
  ).toBeVisible();
  await expect(page.locator(".outcome .status-badge")).toHaveText(
    result.overall_status.replaceAll("_", " "),
  );
  if (
    result.exclusions.some((item: { triggered: boolean }) => item.triggered)
  ) {
    await expect(
      page.getByRole("heading", { name: "Blocking policy findings" }),
    ).toBeVisible();
  }
  await page.screenshot({
    path: "test-results/live-uhc-result.png",
    fullPage: true,
  });
  console.log(
    `Live UHC: HTTP ${response.status()}, ${result.overall_status}, ${result.criteria.length} criteria, ${result.exclusions.filter((item: { triggered: boolean }) => item.triggered).length} triggered exclusions.`,
  );
});
