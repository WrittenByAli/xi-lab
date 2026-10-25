const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json()
}

export type Gender = 'male' | 'female'
export type PitchCondition =
  | 'auto'
  | 'balanced'
  | 'batting_friendly'
  | 'bowling_friendly'
  | 'spin_friendly'
  | 'seam_friendly'
export type TossDecision = 'bat' | 'bowl'

export interface PlayerIn {
  player_id: string
  name: string
  role: string
  bat_rating: number
  bowl_rating: number
  is_keeper: boolean
  is_overseas: boolean
  bat_hand: string
  bowl_type: string
}

export interface PlayerSummary {
  player_id: string
  name: string
  role: string
  bat_rating: number
  bowl_rating: number
  is_keeper: boolean
  is_overseas: boolean
}

export interface RecommendResponse {
  xi: PlayerSummary[]
  win_prob: number
  batting_order: string[]
  bowling_plan: Record<string, number>
  tie_set: string[][]
  swap_matrix: Record<string, Record<string, number>>
}

export interface ScoreDistribution {
  mean: number
  std: number
  median: number
  q10: number
  q90: number
}

export interface SimulateResponse {
  team1_win_prob: number
  team2_win_prob: number
  tie_prob: number
  team1_scores: ScoreDistribution
  team2_scores: ScoreDistribution
}

export interface PhaseRating {
  phase: string
  bat_sr: number | null
  bowl_econ: number | null
  balls: number
}

export interface PlayerProfile {
  player_id: string
  name: string
  role: string
  bat_hand: string | null
  bowl_type: string | null
  is_keeper: boolean
  bat_rating: number | null
  bowl_rating: number | null
  bat_sr: number | null
  bat_avg: number | null
  bowl_econ: number | null
  bowl_sr: number | null
  phase_ratings: PhaseRating[]
  total_balls_bat: number
  total_balls_bowl: number
}

export interface PlayerSearchResult {
  player_id: string
  name: string
  bat_hand: string | null
  bowl_type: string | null
  is_keeper: boolean
}

export interface BacktestMetrics {
  format: string
  test_matches: number
  brier_score: number | null
  log_loss: number | null
  accuracy: number | null
  kelly_roi: number | null
  calibration: { bin_center: number; predicted_mean: number; actual_mean: number; count: number }[]
}

export interface TeamSummary {
  team: string
  recent_matches: number
  squad_size: number
}

export interface SquadPlayer {
  player_id: string
  name: string
  role: string
  bat_rating: number
  bowl_rating: number
  bat_hand: string
  bowl_type: string
  is_keeper: boolean
  recent_matches: number
}

export interface TeamSquad {
  team: string
  format: string
  gender: Gender
  all_time: boolean
  squad_size: number
  players: SquadPlayer[]
}

export interface Venue {
  venue_id: string
  name: string
  match_count: number
}

export interface CompareResponse {
  delta_wp: number
  ci_lower: number
  ci_upper: number
  team_a_wp: number
  team_b_wp: number
}

export interface MatchContext {
  venue?: string | null
  pitchCondition?: PitchCondition
  tossDecision?: TossDecision | null
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  recommend: (
    squad: PlayerIn[],
    format = 't20',
    gender: Gender = 'male',
    opponentTeam?: string | null,
    ctx: MatchContext = {},
    maxOverseas = 4
  ) =>
    request<RecommendResponse>('/recommend', {
      method: 'POST',
      body: JSON.stringify({
        squad,
        format,
        gender,
        opponent_team: opponentTeam || null,
        venue: ctx.venue || null,
        pitch_condition: ctx.pitchCondition ?? 'auto',
        toss_decision: ctx.tossDecision ?? null,
        max_overseas: maxOverseas,
      }),
    }),

  simulate: (
    team1: PlayerIn[],
    team2: PlayerIn[],
    format = 't20',
    gender: Gender = 'male',
    ctx: MatchContext = {},
    nSims = 5000
  ) =>
    request<SimulateResponse>('/simulate', {
      method: 'POST',
      body: JSON.stringify({
        team1,
        team2,
        format,
        gender,
        venue: ctx.venue || null,
        pitch_condition: ctx.pitchCondition ?? 'auto',
        toss_decision: ctx.tossDecision ?? null,
        n_sims: nSims,
      }),
    }),

  searchPlayers: (q: string, limit = 20) =>
    request<PlayerSearchResult[]>(`/player/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  getPlayer: (id: string, format = 't20', gender: Gender = 'male') =>
    request<PlayerProfile>(
      `/player/${encodeURIComponent(id)}?format=${format}&gender=${gender}`
    ),

  getResults: (format = 't20', gender: Gender = 'male') =>
    request<BacktestMetrics>(`/results?format=${format}&gender=${gender}`),

  getAllMetrics: (format = 't20', gender: Gender = 'male') =>
    request<Record<string, unknown>>(`/results/all?format=${format}&gender=${gender}`),

  listTeams: (format = 't20', gender: Gender = 'male', limit = 50) =>
    request<TeamSummary[]>(`/teams?format=${format}&gender=${gender}&limit=${limit}`),

  getSquad: (team: string, format = 't20', gender: Gender = 'male', allTime = false, months = 12) =>
    request<TeamSquad>(
      `/teams/${encodeURIComponent(team)}/squad?format=${format}&gender=${gender}` +
        (allTime ? '&all_time=true' : `&months=${months}`)
    ),

  getVenues: (format = 't20', gender: Gender = 'male') =>
    request<Venue[]>(`/teams/venues?format=${format}&gender=${gender}`),

  compare: (
    teamA: PlayerIn[],
    teamB: PlayerIn[],
    opponent: PlayerIn[],
    format = 't20',
    gender: Gender = 'male',
    ctx: MatchContext = {},
    nSims = 10000
  ) =>
    request<CompareResponse>('/compare', {
      method: 'POST',
      body: JSON.stringify({
        team1_a: teamA,
        team1_b: teamB,
        opponent,
        format,
        gender,
        venue: ctx.venue || null,
        pitch_condition: ctx.pitchCondition ?? 'auto',
        n_sims: nSims,
      }),
    }),
}
