import { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GitCompareArrows, Plus, X, Users, Search, ArrowRight } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Skeleton from '../components/ui/Skeleton'
import { api } from '../lib/api'
import type { PlayerIn, PlayerSearchResult, CompareResponse, TeamSummary, SquadPlayer } from '../lib/api'

function squadPlayerToPlayerIn(p: SquadPlayer): PlayerIn {
  return {
    player_id: p.player_id, name: p.name, role: p.role,
    bat_rating: p.bat_rating, bowl_rating: p.bowl_rating,
    is_keeper: p.is_keeper, is_overseas: false,
    bat_hand: p.bat_hand ?? 'R', bowl_type: p.bowl_type ?? 'none',
  }
}

function makePlayerIn(p: PlayerSearchResult): PlayerIn {
  return {
    player_id: p.player_id, name: p.name, role: 'pure_bat',
    bat_rating: 0.5, bowl_rating: 0.5, is_keeper: p.is_keeper,
    is_overseas: false, bat_hand: p.bat_hand ?? 'R', bowl_type: p.bowl_type ?? 'none',
  }
}

function TeamPanel({ label, color, team, format, onAdd, onRemove, onLoadTeam }: {
  label: string; color: string; team: PlayerIn[]; format: string
  onAdd: (p: PlayerSearchResult) => void; onRemove: (id: string) => void
  onLoadTeam: (players: PlayerIn[]) => void
}) {
  const [q, setQ] = useState('')
  const [results, setResults] = useState<PlayerSearchResult[]>([])
  const [showPicker, setShowPicker] = useState(false)
  const [teams, setTeams] = useState<TeamSummary[]>([])
  const [teamsLoading, setTeamsLoading] = useState(false)
  const [teamFilter, setTeamFilter] = useState('')
  const [loadingSquad, setLoadingSquad] = useState('')
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    if (showPicker && teams.length === 0) {
      setTeamsLoading(true)
      api.listTeams(format, 80).then(setTeams).catch(() => {}).finally(() => setTeamsLoading(false))
    }
  }, [showPicker, format, teams.length])

  const search = useCallback((val: string) => {
    setQ(val)
    if (timer.current) clearTimeout(timer.current)
    if (val.length < 2) { setResults([]); return }
    timer.current = setTimeout(async () => {
      try {
        const r = await api.searchPlayers(val)
        setResults(r.filter(p => !team.some(t => t.player_id === p.player_id)))
      } catch (_) { setResults([]) }
    }, 200)
  }, [team])

  const loadSquad = async (teamName: string) => {
    setLoadingSquad(teamName)
    try {
      const data = await api.getSquad(teamName, format)
      onLoadTeam(data.players.slice(0, 11).map(squadPlayerToPlayerIn))
      setShowPicker(false)
    } catch (_) { /* */ }
    setLoadingSquad('')
  }

  const filtered = teamFilter
    ? teams.filter(t => t.team.toLowerCase().includes(teamFilter.toLowerCase()))
    : teams

  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
        <h3 className="text-[12px] font-semibold uppercase tracking-wider text-text-secondary">{label}</h3>
        <span className="stat text-[11px] text-text-tertiary ml-auto">{team.length}/11</span>
      </div>

      <div className="flex gap-1.5 mb-1.5">
        <button
          onClick={() => setShowPicker(!showPicker)}
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-surface-2 border border-surface-border
                     hover:border-accent/30 hover:bg-surface-3 transition-all cursor-pointer text-[11px] font-medium text-text-primary shrink-0"
        >
          <Users className="w-3 h-3 text-accent" />
          Load
        </button>
        <div className="relative flex-1">
          <Input
            icon placeholder="Add player..."
            value={q} onChange={(e) => search(e.target.value)}
          />
          {results.length > 0 && (
            <div className="absolute z-20 top-full mt-1 w-full bg-surface-2 border border-surface-border rounded-xl shadow-2xl shadow-black/40 overflow-hidden max-h-40 overflow-y-auto">
              {results.slice(0, 6).map((p) => (
                <div
                  key={p.player_id}
                  onClick={() => { onAdd(p); setQ(''); setResults([]) }}
                  className="flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:bg-surface-3 transition-colors text-[12px] text-text-primary"
                >
                  <Plus className="w-3 h-3 text-accent" /> {p.name}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <AnimatePresence>
        {showPicker && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden mb-1.5"
          >
            <div className="bg-surface-2 border border-surface-border rounded-xl p-2.5">
              <div className="relative mb-1.5">
                <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-text-tertiary" />
                <input
                  type="text"
                  placeholder="Filter teams..."
                  value={teamFilter}
                  onChange={(e) => setTeamFilter(e.target.value)}
                  className="w-full pl-7 pr-3 py-1.5 rounded-lg bg-surface-3 border border-surface-border text-[11px]
                             text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/40"
                />
              </div>
              <div className="max-h-36 overflow-y-auto space-y-0.5">
                {teamsLoading ? (
                  Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-7 rounded-lg" />)
                ) : filtered.slice(0, 25).map((t) => (
                  <button
                    key={t.team}
                    onClick={() => loadSquad(t.team)}
                    disabled={loadingSquad === t.team}
                    className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg
                               hover:bg-surface-3 transition-colors cursor-pointer text-left text-[11px]"
                  >
                    <span className="text-text-primary font-medium">{t.team}</span>
                    <span className="text-text-tertiary">{t.recent_matches}m</span>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="space-y-0.5 max-h-[280px] overflow-y-auto">
        {team.map((p, i) => (
          <div key={p.player_id} className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-surface-2/60 group">
            <span className="stat text-[10px] text-text-tertiary w-3 text-right">{i + 1}</span>
            <span className="text-[12px] text-text-primary truncate flex-1">{p.name}</span>
            <button onClick={() => onRemove(p.player_id)} className="opacity-0 group-hover:opacity-100 p-0.5 cursor-pointer">
              <X className="w-3 h-3 text-text-tertiary hover:text-danger" />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

const FORMATS = ['t20', 'odi'] as const

export default function Compare() {
  const [teamA, setTeamA] = useState<PlayerIn[]>([])
  const [teamB, setTeamB] = useState<PlayerIn[]>([])
  const [opponent, setOpponent] = useState<PlayerIn[]>([])
  const [format, setFormat] = useState<string>('t20')
  const [result, setResult] = useState<CompareResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const canCompare = teamA.length === 11 && teamB.length === 11 && opponent.length === 11

  const runCompare = async () => {
    if (!canCompare) return
    setLoading(true)
    try {
      const r = await api.compare(teamA, teamB, opponent, format)
      setResult(r)
    } catch (_) { /* */ }
    setLoading(false)
  }

  const deltaColor = result
    ? result.delta_wp > 0 ? 'text-green-400' : result.delta_wp < 0 ? 'text-red-400' : 'text-text-secondary'
    : ''

  return (
    <div className="pb-12">
      <PageHeader
        title="Head-to-Head Compare"
        description="Compare two XI variants against the same opponent — see the win probability delta"
      >
        <div className="flex items-center gap-3">
          <div className="flex bg-surface-2 rounded-lg p-0.5">
            {FORMATS.map((f) => (
              <button
                key={f}
                onClick={() => setFormat(f)}
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
          <Button onClick={runCompare} disabled={!canCompare || loading}>
            <GitCompareArrows className="w-4 h-4" />
            {loading ? 'Comparing...' : 'Compare'}
          </Button>
        </div>
      </PageHeader>

      <div className="px-8 space-y-5">
        <div className="grid grid-cols-3 gap-4">
          <Card padding="md">
            <TeamPanel
              label="XI Variant A" color="#e5a84b" team={teamA} format={format}
              onAdd={(p) => teamA.length < 11 && setTeamA(prev => [...prev, makePlayerIn(p)])}
              onRemove={(id) => setTeamA(prev => prev.filter(p => p.player_id !== id))}
              onLoadTeam={setTeamA}
            />
          </Card>
          <Card padding="md">
            <TeamPanel
              label="XI Variant B" color="#60a5fa" team={teamB} format={format}
              onAdd={(p) => teamB.length < 11 && setTeamB(prev => [...prev, makePlayerIn(p)])}
              onRemove={(id) => setTeamB(prev => prev.filter(p => p.player_id !== id))}
              onLoadTeam={setTeamB}
            />
          </Card>
          <Card padding="md">
            <TeamPanel
              label="Opponent" color="#a78bfa" team={opponent} format={format}
              onAdd={(p) => opponent.length < 11 && setOpponent(prev => [...prev, makePlayerIn(p)])}
              onRemove={(id) => setOpponent(prev => prev.filter(p => p.player_id !== id))}
              onLoadTeam={setOpponent}
            />
          </Card>
        </div>

        <AnimatePresence>
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <Skeleton className="h-52 rounded-xl" />
            </motion.div>
          )}

          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Main delta result */}
              <Card>
                <div className="flex items-center justify-center gap-8 py-4">
                  <div className="text-center">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">Variant A</div>
                    <div className="stat text-2xl font-bold text-role-bat">
                      {(result.team_a_wp * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="flex flex-col items-center">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">Delta WP</div>
                    <motion.div
                      initial={{ scale: 0.8 }}
                      animate={{ scale: 1 }}
                      className={`stat text-4xl font-bold ${deltaColor}`}
                    >
                      {result.delta_wp > 0 ? '+' : ''}{result.delta_wp.toFixed(2)}
                    </motion.div>
                    <div className="text-[11px] text-text-tertiary mt-1">
                      runs advantage
                    </div>
                  </div>

                  <div className="text-center">
                    <div className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">Variant B</div>
                    <div className="stat text-2xl font-bold text-role-bowl">
                      {(result.team_b_wp * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Confidence interval bar */}
                <div className="mt-4 px-8">
                  <div className="flex justify-between text-[10px] text-text-tertiary mb-1">
                    <span>95% Confidence Interval</span>
                    <span className="stat">[{result.ci_lower.toFixed(2)}, {result.ci_upper.toFixed(2)}]</span>
                  </div>
                  <div className="relative h-8 rounded-lg bg-surface-3 overflow-hidden">
                    {/* Center line at 0 */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-text-tertiary/30" />

                    {/* CI range */}
                    {(() => {
                      const range = Math.max(Math.abs(result.ci_lower), Math.abs(result.ci_upper), 5) * 2
                      const left = ((result.ci_lower + range / 2) / range) * 100
                      const right = ((result.ci_upper + range / 2) / range) * 100
                      const point = ((result.delta_wp + range / 2) / range) * 100
                      return (
                        <>
                          <motion.div
                            className="absolute top-2 bottom-2 rounded-md bg-accent/25"
                            initial={{ left: '50%', width: 0 }}
                            animate={{ left: `${left}%`, width: `${right - left}%` }}
                            transition={{ duration: 0.5, ease: 'easeOut' }}
                          />
                          <motion.div
                            className="absolute top-1 bottom-1 w-1 rounded-full bg-accent"
                            initial={{ left: '50%' }}
                            animate={{ left: `${point}%` }}
                            transition={{ duration: 0.6, ease: 'easeOut' }}
                          />
                        </>
                      )
                    })()}
                  </div>
                  <div className="flex justify-between text-[10px] text-text-tertiary mt-1">
                    <span>B better <ArrowRight className="w-3 h-3 inline rotate-180" /></span>
                    <span><ArrowRight className="w-3 h-3 inline" /> A better</span>
                  </div>
                </div>
              </Card>

              {/* Interpretation */}
              <Card>
                <div className="text-[13px] text-text-secondary leading-relaxed">
                  {result.delta_wp > 0 ? (
                    <>
                      <span className="font-medium text-green-400">Variant A</span> scores on average{' '}
                      <span className="stat font-semibold text-text-primary">{Math.abs(result.delta_wp).toFixed(2)}</span> more runs
                      per innings than <span className="font-medium text-role-bowl">Variant B</span> against this opponent.
                      {result.ci_lower > 0 && (
                        <span className="text-text-tertiary"> This difference is statistically significant (95% CI excludes zero).</span>
                      )}
                      {result.ci_lower <= 0 && (
                        <span className="text-text-tertiary"> However, the confidence interval includes zero — the difference may not be significant.</span>
                      )}
                    </>
                  ) : result.delta_wp < 0 ? (
                    <>
                      <span className="font-medium text-role-bowl">Variant B</span> scores on average{' '}
                      <span className="stat font-semibold text-text-primary">{Math.abs(result.delta_wp).toFixed(2)}</span> more runs
                      per innings than <span className="font-medium text-role-bat">Variant A</span> against this opponent.
                      {result.ci_upper < 0 && (
                        <span className="text-text-tertiary"> This difference is statistically significant (95% CI excludes zero).</span>
                      )}
                      {result.ci_upper >= 0 && (
                        <span className="text-text-tertiary"> However, the confidence interval includes zero — the difference may not be significant.</span>
                      )}
                    </>
                  ) : (
                    <span>Both variants perform equally against this opponent.</span>
                  )}
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
