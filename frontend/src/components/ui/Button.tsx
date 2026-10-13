import { forwardRef } from 'react'
import clsx from 'clsx'

type Variant = 'primary' | 'secondary' | 'ghost'

const variants: Record<Variant, string> = {
  primary:
    'bg-accent text-text-inverse font-semibold hover:bg-accent-hover active:scale-[0.98] shadow-sm shadow-accent/20',
  secondary:
    'bg-surface-3 text-text-primary border border-surface-border hover:bg-surface-4 hover:border-surface-4',
  ghost:
    'text-text-secondary hover:text-text-primary hover:bg-surface-3/50',
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: 'sm' | 'md'
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={clsx(
          'inline-flex items-center justify-center gap-2 rounded-lg transition-all duration-150 cursor-pointer',
          'disabled:opacity-40 disabled:pointer-events-none',
          variants[variant],
          size === 'sm' ? 'px-3 py-1.5 text-[12px]' : 'px-4 py-2.5 text-[13px]',
          className
        )}
        {...props}
      >
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'
export default Button
