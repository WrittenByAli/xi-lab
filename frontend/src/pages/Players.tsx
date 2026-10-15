import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'
import Input from '../components/ui/Input'
import Card from '../components/ui/Card'
import Badge, { roleBadgeVariant, roleLabel } from '../components/ui/Badge'
import Skeleton from '../components/ui/Skeleton'
import { api } from '../lib/api'
import type { PlayerSearchResult, PlayerProfile } from '../lib/api'

export default function Players() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PlayerSearchResult[]>([])
  const [selected, setSelected] = useState<PlayerProfile | null>(null)
  const [searching, setSearching] = useState(false)
  const [loadingProfile, setLoadingProfile] = useState(false)
  const debounce = useRef<ReturnType<typeof setTimeout>>(undefined)

  const search = useCallback((q: string) => {
    setQuery(q)
    if (debounce.current) clearTimeout(debounce.current)
    if (q.length < 2) { setResults([]); return }
    debounce.current = setTimeout(async () => {
      setSearching(true)
      try {
        const r = await api.searchPlayers(q)
        setResults(r)
      } catch (_) { setResults([]) }
      setSearching(false)
    }, 250)
  }, [])

  const [format, setFormat] = useState<string>('t20')

  const selectPlayer = async (id: string, fmt?: string) => {
    setLoadingProfile(true)
    try {
      const p = await api.getPlayer(id, fmt ?? format)
      setSelected(p)
    } catch (_) { /* */ }
    setLoadingProfile(false)
  }

  const switchFormat = (f: string) => {
    setFormat(f)
    if (selected) selectPlayer(selected.player_id, f)
  }

  return (
    <div className="pb-12">
      <PageHeader
        title="Player Explorer"
        description="Search and explore player profiles, ratings, and phase breakdowns"
      >
        <div className="flex bg-surface-2 rounded-lg p-0.5">
          {(['t20', 'odi', 'test'] as const).map((f) => (
            <button
              key={f}
              onClick={() => switchFormat(f)}
              className={`
                px-3 py-1.5 rounded-md text-[12px] font-medium uppercase tracking-wider
                transition-all duration-150 cursor-pointer
                ${format === f
                  ? 'bg-surface-3 text-accent shadow-sm'
                  : 'text-text-tertiary hover:text-text-secondary'}
              `}
            >
              {f}
            </button>
          ))}
        </div>
      </PageHeader>

      <div className="px-8 flex gap-6">
        {/* Search panel */}
        <div className="w-[340px] shrink-0 space-y-3">
          <Input
            icon
            placeholder="Search players..."
            value={query}
            onChange={(e) => search(e.target.value)}
            autoFocus
          />

          <div className="space-y-1.5">
            {searching && Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-14 rounded-lg" />
            ))}

            <AnimatePresence mode="popLayout">
              {results.map((p) => (
                <motion.div
                  key={p.player_id}
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  layout
                >
                  <div
                    onClick={() => selectPlayer(p.player_id)}
                    className={`
                      flex items-center gap-3 px-3.5 py-3 rounded-lg cursor-pointer
                      border border-transparent transition-all duration-150
                      ${selected?.player_id === p.player_id
                        ? 'bg-accent/8 border-accent/20'
                        : 'hover:bg-surface-2'}
                    `}
                  >
                    <div className="w-8 h-8 rounded-md bg-surface-3 flex items-center justify-center shrink-0">
                      <span className="text-[12px] font-bold text-text-secondary">
                        {p.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
                      </span>
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] font-medium text-text-primary truncate">{p.name}</div>
                      <div className="flex items-center gap-1 mt-0.5">
                        {p.bat_hand && (
                          <span className="text-[11px] text-text-tertiary">{p.bat_hand}HB</span>
                        )}
                        {p.bowl_type && p.bowl_type !== 'none' && (
                          <span className="text-[11px] text-text-tertiary">· {p.bowl_type}</span>
                        )}
                        {p.is_keeper && <Badge variant="keeper" className="ml-1">WK</Badge>}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {!searching && query.length >= 2 && results.length === 0 && (
              <div className="text-center py-8 text-[13px] text-text-tertiary">
                No players found
              </div>
            )}
          </div>
        </div>

        {/* Profile panel */}
        <div className="flex-1 min-w-0">
          <AnimatePresence mode="wait">
            {loadingProfile ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-3"
              >
                <Skeleton className="h-32 rounded-xl" />
                <Skeleton className="h-48 rounded-xl" />
              </motion.div>
            ) : selected ? (
              <motion.div
                key={selected.player_id}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.2 }}
                className="space-y-4"
              >
                {/* Header card */}
                <Card>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-14 h-14 rounded-xl bg-surface-3 flex items-center justify-center">
                        <span className="text-xl font-bold text-text-secondary">
                          {selected.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
                        </span>
                      </div>
                      <div>
                        <h2 className="text-lg font-semibold text-text-primary">{selected.name}</h2>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant={roleBadgeVariant(selected.role)}>{roleLabel(selected.role)}</Badge>
                          {selected.bat_hand && (
                            <span className="text-[12px] text-text-tertiary">{selected.bat_hand === 'R' ? 'Right' : 'Left'}-hand bat</span>
                          )}
                          {selected.bowl_type && selected.bowl_type !== 'none' && (
                            <span className="text-[12px] text-text-tertiary">· {selected.bowl_type}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <button onClick={() => setSelected(null)} className="p-1.5 rounded-md hover:bg-surface-3 transition-colors cursor-pointer">
                      <X className="w-4 h-4 text-text-tertiary" />
                    </button>
                  </div>
                </Card>

                {/* Ratings */}
                {(selected.bat_rating != null || selected.bowl_rating != null) && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl bg-role-bat/8 border border-role-bat/15 p-4">
                      <div className="text-[10px] uppercase tracking-wider text-role-bat/80">Bat Rating</div>
                      <div className="stat text-2xl font-bold text-role-bat mt-1">
                        {selected.bat_rating?.toFixed(1) ?? '—'}
                      </div>
                    </div>
                    <div className="rounded-xl bg-role-bowl/8 border border-role-bowl/15 p-4">
                      <div className="text-[10px] uppercase tracking-wider text-role-bowl/80">Bowl Rating</div>
                      <div className="stat text-2xl font-bold text-role-bowl mt-1">
                        {selected.bowl_rating?.toFixed(1) ?? '—'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Stats grid */}
                <div className="grid grid-cols-4 gap-3">
                  <div className="rounded-xl border border-surface-border bg-surface-1 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Strike Rate</div>
                    <div className="stat text-xl font-semibold text-role-bat mt-1">
                      {selected.bat_sr?.toFixed(1) ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl border border-surface-border bg-surface-1 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Average</div>
                    <div className="stat text-xl font-semibold text-text-primary mt-1">
                      {selected.bat_avg?.toFixed(1) ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl border border-surface-border bg-surface-1 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Economy</div>
                    <div className="stat text-xl font-semibold text-role-bowl mt-1">
                      {selected.bowl_econ?.toFixed(2) ?? '—'}
                    </div>
                  </div>
                  <div className="rounded-xl border border-surface-border bg-surface-1 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Bowl SR</div>
                    <div className="stat text-xl font-semibold text-text-primary mt-1">
                      {selected.bowl_sr?.toFixed(1) ?? '—'}
                    </div>
                  </div>
                </div>

                {/* Phase breakdown */}
                {selected.phase_ratings && selected.phase_ratings.length > 0 && (
                  <Card>
                    <h3 className="text-[13px] font-medium uppercase tracking-[0.06em] text-text-secondary mb-4">
                      Phase Breakdown
                    </h3>
                    <div className="space-y-2">
                      {selected.phase_ratings.map((pr) => (
                        <div key={pr.phase} className="flex items-center gap-4">
                          <span className="w-24 text-[12px] font-medium text-text-tertiary uppercase tracking-wider">
                            {pr.phase}
                          </span>
                          <div className="flex-1 flex items-center gap-4">
                            {/* Bat SR bar */}
                            <div className="flex-1">
                              <div className="flex justify-between text-[11px] mb-1">
                                <span className="text-text-tertiary">SR</span>
                                <span className="stat text-role-bat">{pr.bat_sr?.toFixed(1) ?? '—'}</span>
                              </div>
                              <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
                                <motion.div
                                  className="h-full rounded-full bg-role-bat/70"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.min(100, (pr.bat_sr ?? 0) / 2)}%` }}
                                  transition={{ duration: 0.5, ease: 'easeOut' }}
                                />
                              </div>
                            </div>
                            {/* Bowl econ bar */}
                            <div className="flex-1">
                              <div className="flex justify-between text-[11px] mb-1">
                                <span className="text-text-tertiary">Econ</span>
                                <span className="stat text-role-bowl">{pr.bowl_econ?.toFixed(2) ?? '—'}</span>
                              </div>
                              <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
                                <motion.div
                                  className="h-full rounded-full bg-role-bowl/70"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.min(100, (pr.bowl_econ ?? 0) * 8)}%` }}
                                  transition={{ duration: 0.5, ease: 'easeOut' }}
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {/* Volume */}
                <div className="flex gap-3">
                  <Card className="flex-1">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Balls Batted</div>
                    <div className="stat text-lg font-semibold text-text-primary mt-1">
                      {selected.total_balls_bat.toLocaleString()}
                    </div>
                  </Card>
                  <Card className="flex-1">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Balls Bowled</div>
                    <div className="stat text-lg font-semibold text-text-primary mt-1">
                      {selected.total_balls_bowl.toLocaleString()}
                    </div>
                  </Card>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center justify-center h-80 text-[13px] text-text-tertiary"
              >
                Search for a player to view their profile
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
