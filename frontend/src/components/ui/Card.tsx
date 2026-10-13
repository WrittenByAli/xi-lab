import clsx from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
  hover?: boolean
}

const paddings = {
  none: '',
  sm: 'p-3.5',
  md: 'p-5',
  lg: 'p-6',
}

export default function Card({ children, className, padding = 'md', hover }: CardProps) {
  return (
    <div
      className={clsx(
        'rounded-xl border border-surface-border bg-surface-1',
        paddings[padding],
        hover && 'transition-colors duration-150 hover:border-surface-4 hover:bg-surface-2/60',
        className
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('flex items-center justify-between mb-4', className)}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <h3 className={clsx('text-[13px] font-medium uppercase tracking-[0.06em] text-text-secondary', className)}>
      {children}
    </h3>
  )
}
