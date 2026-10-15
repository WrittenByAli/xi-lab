import { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Zap, GripVertical, Users, ChevronDown, X, Search, ArrowRightLeft } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'
import Card, { CardTitle } from '../components/ui/Card'
import Badge, { roleBadgeVariant, roleLabel } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Skeleton from '../components/ui/Skeleton'
import PlayerCard from '../components/cricket/PlayerCard'
import { api } from '../lib/api'
import type { PlayerIn, PlayerSearchResult, RecommendResponse, TeamSummary, SquadPlayer } from '../lib/api'

function makePlayerIn(p: PlayerSearchResult): PlayerIn {
  return {
    player_id: p.player_id,
    name: p.name,
    role: 'pure_bat',
    bat_rating: 0.5,
    bowl_rating: 0.5,
    is_keeper: p.is_keeper,
    is_overseas: false,
    bat_hand: p.bat_hand ?? 'R',
    bowl_type: p.bowl_type ?? 'none',
  }
}

function squadPlayerToPlayerIn(p: SquadPlayer): PlayerIn {
  return {
    player_id: p.player_id,
    name: p.name,
    role: p.role,
    bat_rating: p.bat_rating,
    bowl_rating: p.bowl_rating,
    is_keeper: p.is_keeper,
    is_overseas: false,
    bat_hand: p.bat_hand ?? 'R',
    bowl_type: p.bowl_type ?? 'none',
  }
}

const FORMATS = ['t20', 'odi', 'test'] as const

export default function SquadBuilder() {
  const [squad, setSquad] = useState<PlayerIn[]>([])
  const [format, setFormat] = useState<string>('t20')
  const [searchQ, setSearchQ] = useState('')
  const [searchResults, setSearchResults] = useState<PlayerSearchResult[]>([])
  const [result, setResult] = useState<RecommendResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const debounce = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Team preset state
  const [showTeamPicker, setShowTeamPicker] = useState(false)
  const [teams, setTeams] = useState<TeamSummary[]>([])
  const [teamsLoading, setTeamsLoading] = useState(false)
  const [teamFilter, setTeamFilter] = useState('')
  const [loadingSquad, setLoadingSquad] = useState('')

  useEffect(() => {
    if (showTeamPicker && teams.length === 0) {
      setTeamsLoading(true)
      api.listTeams(format, 100).then(setTeams).catch(() => {}).finally(() => setTeamsLoading(false))
    }
  }, [showTeamPicker, format, teams.length])

  const loadTeamSquad = async (team: string) => {
    setLoadingSquad(team)
    try {
      const data = await api.getSquad(team, format)
      const top = data.players.slice(0, 18)
      setSquad(top.map(squadPlayerToPlayerIn))
      setResult(null)
      setShowTeamPicker(false)
    } catch (_) {
      setError(`Failed to load squad for ${team}`)
    }
    setLoadingSquad('')
  }

  const switchFormat = (f: string) => {
    setFormat(f)
    setTeams([])
    setResult(null)
  }

  const searchPlayers = useCallback((q: string) => {
    setSearchQ(q)
    if (debounce.current) clearTimeout(debounce.current)
    if (q.length < 2) { setSearchResults([]); return }
    debounce.current = setTimeout(async () => {
      try {
        const r = await api.searchPlayers(q)
        setSearchResults(r.filter(p => !squad.some(s => s.player_id === p.player_id)))
      } catch (_) { setSearchResults([]) }
    }, 200)
  }, [squad])

  const addPlayer = (p: PlayerSearchResult) => {
    if (squad.some(s => s.player_id === p.player_id)) return
    setSquad(prev => [...prev, makePlayerIn(p)])
    setSearchQ('')
    setSearchResults([])
  }

  const removePlayer = (id: string) => {
    setSquad(prev => prev.filter(p => p.player_id !== id))
  }

  const recommend = async () => {
    if (squad.length < 11) {
      setError('Need at least 11 players in the squad')
      return
    }
    setError('')
    setLoading(true)
    try {
      const r = await api.recommend(squad, format)
      setResult(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get recommendation')
    }
    setLoading(false)
  }

  const filteredTeams = teamFilter
    ? teams.filter(t => t.team.toLowerCase().includes(teamFilter.toLowerCase()))
    : teams

  return (
    <div className="pb-12">
      <PageHeader
        title="Squad Builder"
        description="Load a team roster or search players, then get the optimal XI"
      >
        <div className="flex items-center gap-3">
          <div className="flex bg-surface-2 rounded-lg p-0.5">
            {FORMATS.map((f) => (
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
          <Button onClick={recommend} disabled={squad.length < 11 || loading}>
            <Zap className="w-4 h-4" />
            {loading ? 'Computing...' : 'Get Best XI'}
          </Button>
        </div>
      </PageHeader>

      <div className="px-8 flex gap-6">
        {/* Left: Squad management */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Team preset + search row */}
          <div className="flex gap-3">
            <button
              onClick={() => setShowTeamPicker(true)}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-surface-2 border border-surface-border
                         hover:border-accent/30 hover:bg-surface-3 transition-all cursor-pointer shrink-0"
            >
              <Users className="w-4 h-4 text-accent" />
              <span className="text-[13px] font-medium text-text-primary">Load Team</span>
              <ChevronDown className="w-3.5 h-3.5 text-text-tertiary" />
            </button>
            <div className="relative flex-1">
              <Input
                icon
                placeholder="Search players to add..."
                value={searchQ}
                onChange={(e) => searchPlayers(e.target.value)}
              />
              <AnimatePresence>
                {searchResults.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="absolute z-20 top-full mt-1 w-full bg-surface-2 border border-surface-border rounded-xl shadow-2xl shadow-black/40 overflow-hidden max-h-64 overflow-y-auto"
                  >
                    {searchResults.slice(0, 8).map((p) => (
                      <div
                        key={p.player_id}
                        onClick={() => addPlayer(p)}
                        className="flex items-center gap-3 px-4 py-2.5 cursor-pointer hover:bg-surface-3 transition-colors"
                      >
                        <Plus className="w-3.5 h-3.5 text-accent shrink-0" />
                        <div className="min-w-0 flex-1">
                          <div className="text-[13px] text-text-primary truncate">{p.name}</div>
                          <div className="text-[11px] text-text-tertiary">
                            {p.bat_hand}HB{p.bowl_type && p.bowl_type !== 'none' ? ` · ${p.bowl_type}` : ''}
                            {p.is_keeper ? ' · WK' : ''}
                          </div>
                        </div>
                      </div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Squad list */}
          <Card padding="sm">
            <div className="flex items-center justify-between px-2 py-1 mb-2">
              <CardTitle>
                Squad ({squad.length})
              </CardTitle>
              {squad.length > 0 && (
                <button
                  onClick={() => { setSquad([]); setResult(null) }}
                  className="text-[11px] text-text-tertiary hover:text-danger transition-colors cursor-pointer"
                >
                  Clear all
                </button>
              )}
            </div>

            {squad.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-14 h-14 rounded-2xl bg-surface-2 flex items-center justify-center mx-auto mb-4">
                  <Users className="w-6 h-6 text-text-tertiary" />
                </div>
                <div className="text-[13px] text-text-tertiary mb-1">No players yet</div>
                <div className="text-[12px] text-text-tertiary/60">
                  Click <span className="text-accent font-medium">Load Team</span> to import a roster, or search players above
                </div>
              </div>
            ) : (
              <div className="space-y-1">
                <AnimatePresence mode="popLayout">
                  {squad.map((p) => (
                    <motion.div
                      key={p.player_id}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      layout
                      className="overflow-hidden"
                    >
                      <div className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-surface-2/60 group">
                        <GripVertical className="w-3.5 h-3.5 text-text-tertiary/40 shrink-0" />
                        <div className="w-7 h-7 rounded-md bg-surface-3 flex items-center justify-center shrink-0">
                          <span className="text-[10px] font-bold text-text-secondary">
                            {p.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
                          </span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <span className="text-[13px] font-medium text-text-primary truncate block">{p.name}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          {p.is_keeper && <Badge variant="keeper">WK</Badge>}
                          <Badge variant={roleBadgeVariant(p.role)}>{roleLabel(p.role)}</Badge>
                        </div>
                        <button
                          onClick={() => removePlayer(p.player_id)}
                          className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-danger/10 transition-all cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5 text-danger/60" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}
          </Card>

          {error && (
            <div className="text-[12px] text-danger bg-danger/8 border border-danger/20 rounded-lg px-4 py-2.5">
              {error}
            </div>
          )}
        </div>

        {/* Right: Result panel */}
        <div className="w-[380px] shrink-0 space-y-4">
          {loading ? (
            <>
              <Skeleton className="h-20 rounded-xl" />
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-16 rounded-xl" />
              ))}
            </>
          ) : result ? (
            <motion.div
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-4"
            >
              {/* WP header */}
              <Card className="bg-accent/5 border-accent/15">
                <div className="text-center">
                  <div className="text-[11px] uppercase tracking-wider text-accent/70">Win Probability</div>
                  <div className="stat text-3xl font-bold text-accent mt-1">
                    {(result.win_prob * (result.win_prob > 1 ? 1 : 100)).toFixed(1)}
                    <span className="text-lg ml-0.5">%</span>
                  </div>
                </div>
              </Card>

              {/* Selected XI */}
              <div>
                <CardTitle className="mb-3">Selected XI</CardTitle>
                <div className="space-y-2">
                  {result.xi.map((p, i) => (
                    <PlayerCard key={p.player_id} player={p} rank={i + 1} compact />
                  ))}
                </div>
              </div>

              {/* Batting order */}
              <Card>
                <CardTitle>Batting Order</CardTitle>
                <div className="mt-3 space-y-1">
                  {result.batting_order.map((pid, i) => {
                    const p = result.xi.find(x => x.player_id === pid)
                    return (
                      <div key={pid} className="flex items-center gap-2.5 py-1">
                        <span className="stat text-[11px] text-text-tertiary w-4 text-right">{i + 1}</span>
                        <span className="text-[13px] text-text-primary">{p?.name ?? pid}</span>
                        {p && <Badge variant={roleBadgeVariant(p.role)} className="ml-auto">{roleLabel(p.role)}</Badge>}
                      </div>
                    )
                  })}
                </div>
              </Card>

              {/* Bowling plan */}
              <Card>
                <CardTitle>Bowling Plan</CardTitle>
                <div className="mt-3 space-y-1.5">
                  {Object.entries(result.bowling_plan).map(([pid, overs]) => {
                    const p = result.xi.find(x => x.player_id === pid)
                    return (
                      <div key={pid} className="flex items-center justify-between">
                        <span className="text-[13px] text-text-primary">{p?.name ?? pid}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 rounded-full bg-surface-3 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-role-bowl/60"
                              style={{ width: `${(overs / 4) * 100}%` }}
                            />
                          </div>
                          <span className="stat text-[12px] text-text-secondary w-8 text-right">{overs}ov</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </Card>

              {/* Swap matrix */}
              {result.swap_matrix && Object.keys(result.swap_matrix).length > 0 && (
                <Card>
                  <div className="flex items-center gap-2 mb-3">
                    <ArrowRightLeft className="w-3.5 h-3.5 text-accent" />
                    <CardTitle>Swap Impact</CardTitle>
                  </div>
                  <p className="text-[11px] text-text-tertiary mb-3">
                    Best bench-to-XI swaps by WP change
                  </p>
                  <div className="space-y-1">
                    {(() => {
                      const swaps: { benchId: string; xiId: string; delta: number }[] = []
                      for (const [benchId, targets] of Object.entries(result.swap_matrix)) {
                        for (const [xiId, delta] of Object.entries(targets)) {
                          swaps.push({ benchId, xiId, delta })
                        }
                      }
                      swaps.sort((a, b) => b.delta - a.delta)
                      const topSwaps = swaps.slice(0, 8)
                      if (topSwaps.length === 0) return null

                      const maxAbs = Math.max(...topSwaps.map(s => Math.abs(s.delta)), 0.001)
                      const getName = (id: string) => {
                        const p = squad.find(s => s.player_id === id) ?? result.xi.find(x => x.player_id === id)
                        return p?.name ?? id
                      }

                      return topSwaps.map((s, i) => {
                        const isPositive = s.delta > 0
                        return (
                          <motion.div
                            key={`${s.benchId}-${s.xiId}`}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.03 }}
                            className="flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-surface-2/50"
                          >
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1 text-[11px]">
                                <span className="text-green-400 truncate font-medium">{getName(s.benchId)}</span>
                                <span className="text-text-tertiary shrink-0">for</span>
                                <span className="text-red-400 truncate">{getName(s.xiId)}</span>
                              </div>
                            </div>
                            <div className="w-14 h-1.5 rounded-full bg-surface-3 overflow-hidden shrink-0">
                              <div
                                className={`h-full rounded-full ${isPositive ? 'bg-green-500/60' : 'bg-red-500/40'}`}
                                style={{ width: `${(Math.abs(s.delta) / maxAbs) * 100}%` }}
                              />
                            </div>
                            <span className={`stat text-[11px] w-12 text-right shrink-0 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                              {isPositive ? '+' : ''}{s.delta.toFixed(3)}
                            </span>
                          </motion.div>
                        )
                      })
                    })()}
                  </div>
                </Card>
              )}

              {/* Tie set — near-optimal alternatives */}
              {result.tie_set && result.tie_set.length > 0 && (
                <Card>
                  <CardTitle>Near-Optimal Alternatives</CardTitle>
                  <p className="text-[11px] text-text-tertiary mt-1 mb-3">
                    XIs within the statistical confidence interval
                  </p>
                  <div className="space-y-1.5">
                    {result.tie_set.slice(0, 5).map((alt, i) => {
                      const different = alt.filter(id => !result.xi.some(x => x.player_id === id))
                      const getName = (id: string) => {
                        const p = squad.find(s => s.player_id === id)
                        return p?.name ?? id
                      }
                      return (
                        <div key={i} className="px-2 py-1.5 rounded-md bg-surface-2/40 text-[11px]">
                          <span className="text-text-tertiary">Alt {i + 1}: </span>
                          {different.length === 0 ? (
                            <span className="text-text-secondary">Same as selected XI</span>
                          ) : (
                            <span className="text-accent">
                              +{different.map(getName).join(', ')}
                            </span>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </Card>
              )}
            </motion.div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-12 h-12 rounded-xl bg-surface-2 flex items-center justify-center mb-3">
                <Zap className="w-5 h-5 text-text-tertiary" />
              </div>
              <div className="text-[13px] text-text-tertiary">
                Add 11+ players and hit<br />"Get Best XI" to optimize
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Team picker modal */}
      <AnimatePresence>
        {showTeamPicker && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={(e) => { if (e.target === e.currentTarget) setShowTeamPicker(false) }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              className="w-[480px] max-h-[70vh] bg-surface-1 border border-surface-border rounded-2xl shadow-2xl shadow-black/60 flex flex-col overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-surface-border">
                <div>
                  <h3 className="text-[15px] font-semibold text-text-primary">Load Team Roster</h3>
                  <p className="text-[12px] text-text-tertiary mt-0.5">
                    Select a team to import their recent {format.toUpperCase()} squad
                  </p>
                </div>
                <button
                  onClick={() => setShowTeamPicker(false)}
                  className="p-1.5 rounded-lg hover:bg-surface-3 transition-colors cursor-pointer"
                >
                  <X className="w-4 h-4 text-text-tertiary" />
                </button>
              </div>

              {/* Search */}
              <div className="px-5 py-3 border-b border-surface-border">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-tertiary" />
                  <input
                    type="text"
                    placeholder="Filter teams..."
                    value={teamFilter}
                    onChange={(e) => setTeamFilter(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 rounded-lg bg-surface-2 border border-surface-border text-[13px]
                               text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent/40"
                  />
                </div>
              </div>

              {/* Team list */}
              <div className="flex-1 overflow-y-auto p-2">
                {teamsLoading ? (
                  <div className="space-y-1 p-3">
                    {Array.from({ length: 8 }).map((_, i) => (
                      <Skeleton key={i} className="h-12 rounded-lg" />
                    ))}
                  </div>
                ) : filteredTeams.length === 0 ? (
                  <div className="text-center py-8 text-[13px] text-text-tertiary">
                    No teams found
                  </div>
                ) : (
                  filteredTeams.map((t) => (
                    <button
                      key={t.team}
                      onClick={() => loadTeamSquad(t.team)}
                      disabled={loadingSquad === t.team}
                      className="w-full flex items-center justify-between px-4 py-3 rounded-xl
                                 hover:bg-surface-2 transition-colors cursor-pointer group text-left"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-surface-3 flex items-center justify-center shrink-0">
                          <span className="text-[10px] font-bold text-text-secondary uppercase">
                            {t.team.slice(0, 3)}
                          </span>
                        </div>
                        <div>
                          <div className="text-[13px] font-medium text-text-primary group-hover:text-accent transition-colors">
                            {t.team}
                          </div>
                          <div className="text-[11px] text-text-tertiary">
                            {t.recent_matches} matches · {t.squad_size} players
                          </div>
                        </div>
                      </div>
                      {loadingSquad === t.team ? (
                        <div className="w-4 h-4 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                      ) : (
                        <Plus className="w-4 h-4 text-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity" />
                      )}
                    </button>
                  ))
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
