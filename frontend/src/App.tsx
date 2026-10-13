import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Shell from './components/layout/Shell'
import Dashboard from './pages/Dashboard'
import SquadBuilder from './pages/SquadBuilder'
import Simulator from './pages/Simulator'
import Players from './pages/Players'
import Results from './pages/Results'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, retry: 1 },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<Dashboard />} />
            <Route path="squad" element={<SquadBuilder />} />
            <Route path="simulate" element={<Simulator />} />
            <Route path="players" element={<Players />} />
            <Route path="results" element={<Results />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
