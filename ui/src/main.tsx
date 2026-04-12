import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'sonner'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <App />
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: 'hsl(220 18% 7%)',
          border: '1px solid hsl(220 13% 14%)',
          color: 'hsl(210 40% 98%)',
          fontSize: '13px',
          borderRadius: '10px',
          boxShadow: '0 16px 48px -12px rgba(0, 0, 0, 0.5)',
        },
      }}
    />
  </BrowserRouter>
)
