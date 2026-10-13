import { forwardRef } from 'react'
import clsx from 'clsx'
import { Search } from 'lucide-react'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  icon?: boolean
}

const Input = forwardRef<HTMLInputElement, InputProps>(({ className, icon, ...props }, ref) => {
  if (icon) {
    return (
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary pointer-events-none" />
        <input
          ref={ref}
          className={clsx(
            'w-full bg-surface-2 border border-surface-border rounded-lg pl-9 pr-3.5 py-2.5',
            'text-[13px] text-text-primary placeholder:text-text-tertiary',
            'outline-none transition-colors duration-150',
            'focus:border-accent/40 focus:ring-1 focus:ring-accent/20',
            className
          )}
          {...props}
        />
      </div>
    )
  }

  return (
    <input
      ref={ref}
      className={clsx(
        'w-full bg-surface-2 border border-surface-border rounded-lg px-3.5 py-2.5',
        'text-[13px] text-text-primary placeholder:text-text-tertiary',
        'outline-none transition-colors duration-150',
        'focus:border-accent/40 focus:ring-1 focus:ring-accent/20',
        className
      )}
      {...props}
    />
  )
})

Input.displayName = 'Input'
export default Input
