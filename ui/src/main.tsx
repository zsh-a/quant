import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'sonner'
import { initTheme } from './hooks/useTheme'
import './index.css'
import App from './App.tsx'

// Apply theme before first paint to avoid flash
initTheme()

createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <App />
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: 'var(--color-card)',
          border: '1px solid var(--color-border)',
          color: 'var(--color-foreground)',
          fontSize: '13px',
          borderRadius: '10px',
          boxShadow: '0 16px 48px -12px rgba(0, 0, 0, 0.15)',
        },
      }}
    />
  </BrowserRouter>
)
