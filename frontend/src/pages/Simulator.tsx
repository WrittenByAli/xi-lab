import { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Plus, X, Users, Search } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'
import Card, { CardTitle } from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Skeleton from '../components/ui/Skeleton'
import { api } from '../lib/api'
import type { PlayerIn, PlayerSearchResult, SimulateResponse, TeamSummary, SquadPlayer } from '../lib/api'

function makePlayerIn(p: PlayerSearchResult): PlayerIn {
  return {
    player_id: p.player_id, name: p.name, role: 'pure_bat',
    bat_rating: 0.5, bowl_rating: 0.5, is_keeper: p.is_keeper,
    is_overseas: false, bat_hand: p.bat_hand ?? 'R', bowl_type: p.bowl_type ?? 'none',
  }
}

function squadPlayerToPlayerIn(p: SquadPlayer): PlayerIn {
  return {
    player_id: p.player_id, name: p.name, role: p.role,
    bat_rating: p.bat_rating, bowl_rating: p.bowl_rating,
    is_keeper: p.is_keeper, is_overseas: false,
    bat_hand: p.bat_hand ?? 'R', bowl_type: p.bowl_type ?? 'none',
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
      <div className="flex items-center gap-2 mb-3">
        <div className="w-3 h-3 rounded-full" style={{ background: color }} />
        <h3 className="text-[13px] font-semibold uppercase tracking-wider text-text-secondary">{label}</h3>
        <span className="stat text-[11px] text-text-tertiary ml-auto">{team.length}/11</span>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={() => setShowPicker(!showPicker)}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-surface-2 border border-surface-border
                     hover:border-accent/30 hover:bg-surface-3 transition-all cursor-pointer text-[12px] font-medium text-text-primary shrink-0"
        >
          <Users className="w-3.5 h-3.5 text-accent" />
          Load
        </button>
        <div className="relative flex-1">
          <Input
            icon placeholder={`Add to ${label}...`}
            value={q} onChange={(e) => search(e.target.value)}
          />
          {results.length > 0 && (
            <div className="absolute z-20 top-full mt-1 w-full bg-surface-2 border border-surface-border rounded-xl shadow-2xl shadow-black/40 overflow-hidden max-h-48 overflow-y-auto">
              {results.slice(0, 6).map((p) => (
                <div
                  key={p.player_id}
                  onClick={() => { onAdd(p); setQ(''); setResults([]) }}
                  className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-surface-3 transition-colors text-[13px] text-text-primary"
                >
                  <Plus className="w-3 h-3 text-accent" /> {p.name}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Inline team picker */}
      <AnimatePresence>
        {showPicker && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden mb-2"
          >
            <div className="bg-surface-2 border border-surface-border rounded-xl p-3">
              <div className="relative mb-2">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-text-tertiary" />
                <input
                  type="text"
                  placeholder="Filter teams..."
                  value={teamFilter}
                  onChange={(e) => setTeamFilter(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-surface-3 border border-surface-border text-[12px]
                             text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/40"
                />
              </div>
              <div className="max-h-48 overflow-y-auto space-y-0.5">
                {teamsLoading ? (
                  Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-8 rounded-lg" />)
                ) : filtered.slice(0, 30).map((t) => (
                  <button
                    key={t.team}
                    onClick={() => loadSquad(t.team)}
                    disabled={loadingSquad === t.team}
                    className="w-full flex items-center justify-between px-3 py-2 rounded-lg
                               hover:bg-surface-3 transition-colors cursor-pointer text-left text-[12px]"
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

      <div className="space-y-0.5 max-h-[400px] overflow-y-auto">
        {team.map((p, i) => (
          <div key={p.player_id} className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-surface-2/60 group">
            <span className="stat text-[10px] text-text-tertiary w-4 text-right">{i + 1}</span>
            <span className="text-[13px] text-text-primary truncate flex-1">{p.name}</span>
            <button onClick={() => onRemove(p.player_id)} className="opacity-0 group-hover:opacity-100 p-0.5 cursor-pointer">
              <X className="w-3 h-3 text-text-tertiary hover:text-danger" />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

const FORMATS = ['t20', 'odi', 'test'] as const

export default function Simulator() {
  const [team1, setTeam1] = useState<PlayerIn[]>([])
  const [team2, setTeam2] = useState<PlayerIn[]>([])
  const [format, setFormat] = useState<string>('t20')
  const [result, setResult] = useState<SimulateResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const simulate = async () => {
    if (team1.length !== 11 || team2.length !== 11) return
    setLoading(true)
    try {
      const r = await api.simulate(team1, team2, format, 5000)
      setResult(r)
    } catch (_) { /* */ }
    setLoading(false)
  }

  return (
    <div className="pb-12">
      <PageHeader
        title="Match Simulator"
        description="Pick two XIs and simulate head-to-head"
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
          <Button onClick={simulate} disabled={team1.length !== 11 || team2.length !== 11 || loading}>
            <Play className="w-4 h-4" />
            {loading ? 'Simulating...' : 'Simulate'}
          </Button>
        </div>
      </PageHeader>

      <div className="px-8 space-y-6">
        {/* Team panels */}
        <div className="flex gap-6">
          <Card padding="md" className="flex-1">
            <TeamPanel
              label="Team 1" color="#e5a84b" team={team1} format={format}
              onAdd={(p) => team1.length < 11 && setTeam1(prev => [...prev, makePlayerIn(p)])}
              onRemove={(id) => setTeam1(prev => prev.filter(p => p.player_id !== id))}
              onLoadTeam={setTeam1}
            />
          </Card>
          <Card padding="md" className="flex-1">
            <TeamPanel
              label="Team 2" color="#60a5fa" team={team2} format={format}
              onAdd={(p) => team2.length < 11 && setTeam2(prev => [...prev, makePlayerIn(p)])}
              onRemove={(id) => setTeam2(prev => prev.filter(p => p.player_id !== id))}
              onLoadTeam={setTeam2}
            />
          </Card>
        </div>

        {/* Results */}
        <AnimatePresence>
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <Skeleton className="h-48 rounded-xl" />
            </motion.div>
          )}

          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* WP bar */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <div className="text-center flex-1">
                    <div className="text-[11px] uppercase tracking-wider text-text-tertiary mb-1">Team 1</div>
                    <div className="stat text-3xl font-bold text-role-bat">
                      {(result.team1_win_prob * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-center px-6">
                    <div className="text-[11px] uppercase tracking-wider text-text-tertiary mb-1">Tie</div>
                    <div className="stat text-lg text-text-tertiary">
                      {(result.tie_prob * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-center flex-1">
                    <div className="text-[11px] uppercase tracking-wider text-text-tertiary mb-1">Team 2</div>
                    <div className="stat text-3xl font-bold text-role-bowl">
                      {(result.team2_win_prob * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Visual bar */}
                <div className="h-3 rounded-full overflow-hidden flex bg-surface-3">
                  <motion.div
                    className="bg-role-bat rounded-l-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.team1_win_prob * 100}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut' }}
                  />
                  <motion.div
                    className="bg-surface-4"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.tie_prob * 100}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
                  />
                  <motion.div
                    className="bg-role-bowl rounded-r-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.team2_win_prob * 100}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
                  />
                </div>
              </Card>

              {/* Score distributions */}
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: 'Team 1 Scores', data: result.team1_scores, color: 'text-role-bat' },
                  { label: 'Team 2 Scores', data: result.team2_scores, color: 'text-role-bowl' },
                ].map(({ label, data, color }) => (
                  <Card key={label}>
                    <CardTitle>{label}</CardTitle>
                    <div className="grid grid-cols-3 gap-3 mt-3">
                      {[
                        { k: 'Mean', v: data.mean },
                        { k: 'Median', v: data.median },
                        { k: 'Std Dev', v: data.std },
                        { k: '10th %', v: data.q10 },
                        { k: '90th %', v: data.q90 },
                      ].map(({ k, v }) => (
                        <div key={k}>
                          <div className="text-[10px] uppercase tracking-wider text-text-tertiary">{k}</div>
                          <div className={`stat text-[15px] font-semibold ${color} mt-0.5`}>
                            {v.toFixed(1)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
