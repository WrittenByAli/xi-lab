import { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Shell from './components/layout/Shell'
import Skeleton from './components/ui/Skeleton'
import { GenderProvider } from './lib/genderContext'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const SquadBuilder = lazy(() => import('./pages/SquadBuilder'))
const Simulator = lazy(() => import('./pages/Simulator'))
const Players = lazy(() => import('./pages/Players'))
const Results = lazy(() => import('./pages/Results'))
const Compare = lazy(() => import('./pages/Compare'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, retry: 1 },
  },
})

function PageFallback() {
  return (
    <div className="space-y-4 p-6">
      <Skeleton className="h-8 w-48" />
      <Skeleton className="h-64 w-full" />
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <GenderProvider>
        <BrowserRouter>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route element={<Shell />}>
                <Route index element={<Dashboard />} />
                <Route path="squad" element={<SquadBuilder />} />
                <Route path="simulate" element={<Simulator />} />
                <Route path="players" element={<Players />} />
                <Route path="compare" element={<Compare />} />
                <Route path="results" element={<Results />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </GenderProvider>
    </QueryClientProvider>
  )
}
