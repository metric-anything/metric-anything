import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendIp = env.VITE_BACKEND_IP || "127.0.0.1";

  return {
    server: {
      host: true, // expose on LAN for testing
      port: 5173,
      proxy: {
        "/api/depth": {
          target: `http://${backendIp}:8081`,
          rewrite: (path) => path.replace(/^\/api\/depth/, ""),
          changeOrigin: true,
          ws: true,
        },
        "/api/llm": {
          target: `http://${backendIp}:11434`,
          rewrite: (path) => path.replace(/^\/api\/llm/, ""),
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      rollupOptions: {
        input: {
          main: "index.html",
          chat: "chat.html",
        },
      },
    },
  };
});
