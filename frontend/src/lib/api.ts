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
  squad_size: number
  players: SquadPlayer[]
}

export interface CompareResponse {
  delta_wp: number
  ci_lower: number
  ci_upper: number
  team_a_wp: number
  team_b_wp: number
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  recommend: (squad: PlayerIn[], format = 't20', maxOverseas = 4) =>
    request<RecommendResponse>('/recommend', {
      method: 'POST',
      body: JSON.stringify({ squad, format, max_overseas: maxOverseas }),
    }),

  simulate: (team1: PlayerIn[], team2: PlayerIn[], format = 't20', nSims = 5000) =>
    request<SimulateResponse>('/simulate', {
      method: 'POST',
      body: JSON.stringify({ team1, team2, format, n_sims: nSims }),
    }),

  searchPlayers: (q: string, limit = 20) =>
    request<PlayerSearchResult[]>(`/player/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  getPlayer: (id: string, format = 't20') =>
    request<PlayerProfile>(`/player/${encodeURIComponent(id)}?format=${format}`),

  getResults: (format = 't20') => request<BacktestMetrics>(`/results?format=${format}`),

  getAllMetrics: () => request<Record<string, unknown>>('/results/all'),

  listTeams: (format = 't20', limit = 50) =>
    request<TeamSummary[]>(`/teams?format=${format}&limit=${limit}`),

  getSquad: (team: string, format = 't20', months = 12) =>
    request<TeamSquad>(`/teams/${encodeURIComponent(team)}/squad?format=${format}&months=${months}`),

  compare: (teamA: PlayerIn[], teamB: PlayerIn[], opponent: PlayerIn[], format = 't20', nSims = 10000) =>
    request<CompareResponse>('/compare', {
      method: 'POST',
      body: JSON.stringify({ team1_a: teamA, team1_b: teamB, opponent, format, n_sims: nSims }),
    }),
}
