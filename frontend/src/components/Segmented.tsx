interface Props<T extends string> {
  value: T
  options: readonly T[]
  onChange: (v: T) => void
}

export default function Segmented<T extends string>({
  value,
  options,
  onChange,
}: Props<T>) {
  return (
    <div className="segmented">
      {options.map((opt) => (
        <button
          key={opt}
          className={value === opt ? 'active' : ''}
          onClick={() => onChange(opt)}
          type="button"
        >
          {opt}
        </button>
      ))}
    </div>
  )
}
