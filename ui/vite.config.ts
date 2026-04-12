import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    chunkSizeWarningLimit: 2000,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws/': {
        target: 'ws://localhost:8000',
        ws: true,
        on: {
          error(_err: Error, _req: any, _res: any) { /* silence proxy errors when API is down */ },
          proxyReqWs(_proxyReq: any, _req: any, socket: any) { socket.on('error', () => {}); },
        },
      },
    },
  },
})
