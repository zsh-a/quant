import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { SessionSummary, Trade, Position, EquityPoint } from '../types'

interface SessionData {
  equity: EquityPoint[]
  trades: Trade[]
  positions: Record<string, Position>
}

interface SessionState {
  sessions: SessionSummary[]
  selectedSessionIds: string[]
  primarySessionId: string | null
  sessionDataCache: Record<string, SessionData>
}

type SessionStore = SessionState & {
  actions: {
    setSessions: (sessions: SessionSummary[]) => void
    selectSession: (id: string) => void
    deselectSession: (id: string) => void
    toggleSession: (id: string) => void
    updateSession: (id: string, data: Partial<SessionSummary>) => void
    addSessionData: (id: string, data: SessionData) => void
    removeSession: (id: string) => void
    clearAll: () => void
  }
}

const initialState: SessionState = {
  sessions: [],
  selectedSessionIds: [],
  primarySessionId: null,
  sessionDataCache: {}
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      ...initialState,
      
      actions: {
        setSessions: (sessions) => set({ sessions }),
        
        selectSession: (id) => {
          const { selectedSessionIds } = get()
          if (!selectedSessionIds.includes(id)) {
            set({ selectedSessionIds: [...selectedSessionIds, id] })
          }
          set({ primarySessionId: id })
        },
        
        deselectSession: (id) => {
          const { selectedSessionIds, primarySessionId } = get()
          const newSelected = selectedSessionIds.filter(s => s !== id)
          set({
            selectedSessionIds: newSelected,
            primarySessionId: primarySessionId === id
              ? (newSelected.length > 0 ? newSelected[0] : null)
              : primarySessionId
          })
        },
        
        toggleSession: (id) => {
          const { selectedSessionIds, actions } = get()
          if (selectedSessionIds.includes(id)) {
            actions.deselectSession(id)
          } else {
            actions.selectSession(id)
          }
        },
        
        updateSession: (id, data) => set((state) => ({
          sessions: state.sessions.map(s =>
            s.id === id ? { ...s, ...data } : s
          )
        })),
        
        addSessionData: (id, data) => set((state) => ({
          sessionDataCache: {
            ...(state.sessionDataCache || {}),
            [id]: data
          }
        })),
        
        removeSession: (id) => set((state) => {
          const sessionDataCache = state.sessionDataCache || {}
          const { [id]: _, ...rest } = sessionDataCache
          return {
            sessions: (state.sessions || []).filter(s => s.id !== id),
            selectedSessionIds: (state.selectedSessionIds || []).filter(s => s !== id),
            primarySessionId: state.primarySessionId === id
              ? null
              : state.primarySessionId,
            sessionDataCache: rest
          }
        }),
        
        clearAll: () => set({
          ...initialState
        })
      }
    }),
    {
      name: 'session-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        selectedSessionIds: (state.selectedSessionIds || []).filter((id: string) => typeof id === 'string'),
        primarySessionId: typeof state.primarySessionId === 'string' ? state.primarySessionId : null
      })
    }
  )
)

export const useSessionActions = () => useSessionStore(state => state.actions)
export const useSessions = (): SessionSummary[] => useSessionStore(state => state.sessions || [])
export const useSelectedSessions = (): string[] => useSessionStore(state => state.selectedSessionIds || [])
export const usePrimarySession = (): string | null => useSessionStore(state => state.primarySessionId)
export const useSessionDataCache = (): Record<string, SessionData> => useSessionStore(state => state.sessionDataCache || {})
