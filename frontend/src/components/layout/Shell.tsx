import { NavLink, Outlet } from 'react-router-dom'
import { LayoutDashboard, Users, Swords, UserSearch, BarChart3, Activity, GitCompareArrows } from 'lucide-react'
import clsx from 'clsx'
import { useGender, type Gender } from '../../lib/genderContext'

const links = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/squad', icon: Users, label: 'Squad Builder' },
  { to: '/simulate', icon: Swords, label: 'Simulator' },
  { to: '/compare', icon: GitCompareArrows, label: 'Compare' },
  { to: '/players', icon: UserSearch, label: 'Players' },
  { to: '/results', icon: BarChart3, label: 'Results' },
]

const GENDER_OPTIONS: { value: Gender; label: string }[] = [
  { value: 'male', label: "Men's" },
  { value: 'female', label: "Women's" },
]

function GenderToggle() {
  const { gender, setGender } = useGender()
  return (
    <div className="px-5 pt-3">
      <div className="flex bg-surface-2 rounded-lg p-0.5">
        {GENDER_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setGender(opt.value)}
            className={clsx(
              'flex-1 px-2.5 py-1.5 rounded-md text-[12px] font-medium transition-all duration-150 cursor-pointer',
              gender === opt.value
                ? 'bg-surface-3 text-accent shadow-sm'
                : 'text-text-tertiary hover:text-text-secondary'
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function Shell() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-[220px] shrink-0 border-r border-surface-border bg-surface-1 flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center gap-2.5 px-5 border-b border-surface-border">
          <div className="w-8 h-8 rounded-lg bg-accent/15 flex items-center justify-center">
            <Activity className="w-4 h-4 text-accent" />
          </div>
          <div>
            <span className="text-[15px] font-semibold tracking-tight text-text-primary">XI-Lab</span>
            <span className="block text-[10px] uppercase tracking-[0.1em] text-text-tertiary leading-none mt-0.5">
              Cricket Engine
            </span>
          </div>
        </div>

        {/* Gender */}
        <GenderToggle />

        {/* Nav */}
        <nav className="flex-1 py-3 px-2.5">
          <div className="text-[10px] uppercase tracking-[0.12em] text-text-tertiary px-2.5 mb-2 mt-1">
            Analysis
          </div>
          {links.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] font-medium transition-all duration-150',
                  isActive
                    ? 'bg-accent/10 text-accent'
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface-3/50'
                )
              }
            >
              <Icon className="w-[16px] h-[16px]" strokeWidth={1.8} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3.5 border-t border-surface-border">
          <div className="text-[11px] text-text-tertiary">
            Data: Cricsheet ODC-BY
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto bg-surface-0">
        <Outlet />
      </main>
    </div>
  )
}
