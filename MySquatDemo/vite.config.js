import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: true, // expose on LAN for testing
    port: 5173,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
