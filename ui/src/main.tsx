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
          background: 'hsl(225 14% 9%)',
          border: '1px solid hsl(225 10% 16%)',
          color: 'hsl(40 10% 92%)',
          fontSize: '13px',
          borderRadius: '6px',
        },
      }}
    />
  </BrowserRouter>
)
