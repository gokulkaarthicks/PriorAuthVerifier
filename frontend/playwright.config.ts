import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  use: { baseURL: "http://127.0.0.1:5189", browserName: "chromium" },
  webServer: {
    command: "npm run dev -- --port 5189",
    url: "http://127.0.0.1:5189",
    reuseExistingServer: false,
  },
});
