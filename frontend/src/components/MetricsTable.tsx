import type { ModelMetrics } from '../types'

interface Props {
  metrics: Record<string, ModelMetrics>
  trainingTime: Record<string, number>
  sortKey: string
  ascending?: boolean
  highlight?: string
}

export default function MetricsTable({
  metrics,
  trainingTime,
  sortKey,
  ascending = false,
  highlight,
}: Props) {
  const names = Object.keys(metrics)
  if (names.length === 0) return null
  const cols = Object.keys(metrics[names[0]])
  const sorted = [...names].sort(
    (a, b) =>
      (metrics[a][sortKey] - metrics[b][sortKey]) * (ascending ? 1 : -1),
  )
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          {cols.map((c) => (
            <th key={c}>{c}</th>
          ))}
          <th>Time (s)</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((name) => (
          <tr key={name} style={highlight === name ? { background: '#eef2ff' } : {}}>
            <td>
              <strong>{name}</strong>
            </td>
            {cols.map((c) => (
              <td key={c}>{metrics[name][c]?.toFixed(4)}</td>
            ))}
            <td>{trainingTime[name]?.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
