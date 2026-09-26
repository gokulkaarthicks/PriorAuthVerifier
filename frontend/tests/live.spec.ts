import { expect, test } from "@playwright/test";

// Never part of the default validation; requires explicit permission for provider use.
test("one live UHC job through the queue", async ({ page }) => {
  test.skip(
    process.env.RUN_LIVE_UHC !== "1",
    "Paid large-policy test is opt-in",
  );
  test.setTimeout(15 * 60 * 1000);
  await page.goto("/");
  await page
    .locator('input[aria-label="Patient chart"]')
    .setInputFiles("../sample_data/patient.pdf");
  await page
    .locator('input[aria-label="Policy / guideline"]')
    .setInputFiles("../sample_data/uhc_policy.pdf");
  const created = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/jobs") &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Start verification" }).click();
  expect((await created).status()).toBe(202);
  await expect(
    page.getByRole("button", { name: "Review patient.pdf" }),
  ).toBeVisible({ timeout: 14 * 60 * 1000 });
  await page.getByRole("button", { name: "Review patient.pdf" }).click();
  await expect(
    page.getByRole("heading", {
      name: "Prior authorization review",
      exact: true,
    }),
  ).toBeVisible();
});
