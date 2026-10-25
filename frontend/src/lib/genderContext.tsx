import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

export type Gender = 'male' | 'female'

const STORAGE_KEY = 'xilab_gender'

interface GenderContextValue {
  gender: Gender
  setGender: (g: Gender) => void
}

const GenderContext = createContext<GenderContextValue | undefined>(undefined)

function readStored(): Gender {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'female' ? 'female' : 'male'
  } catch (_) {
    return 'male'
  }
}

export function GenderProvider({ children }: { children: ReactNode }) {
  const [gender, setGenderState] = useState<Gender>(readStored)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, gender)
    } catch (_) {
      /* ignore (private browsing, blocked storage, etc.) */
    }
  }, [gender])

  const setGender = (g: Gender) => setGenderState(g)

  return (
    <GenderContext.Provider value={{ gender, setGender }}>{children}</GenderContext.Provider>
  )
}

export function useGender(): GenderContextValue {
  const ctx = useContext(GenderContext)
  if (!ctx) throw new Error('useGender must be used within a GenderProvider')
  return ctx
}
