import Plot, { baseLayout, baseConfig } from './Plot'
import type { ModelMetrics } from '../types'

interface Props {
  metrics: Record<string, ModelMetrics>
  metricKey: string
  title: string
}

export default function MetricBar({ metrics, metricKey, title }: Props) {
  const items = Object.entries(metrics).sort(
    (a, b) => b[1][metricKey] - a[1][metricKey],
  )
  const names = items.map(([n]) => n)
  const values = items.map(([, m]) => m[metricKey])
  return (
    <Plot
      data={[
        {
          type: 'bar',
          orientation: 'h',
          x: values,
          y: names,
          text: values.map((v) => v.toFixed(4)),
          textposition: 'outside',
          marker: { color: '#1b6cf5' },
          hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
        },
      ]}
      layout={{
        ...baseLayout,
        title,
        xaxis: { title: metricKey },
        yaxis: { autorange: 'reversed' },
        height: 380,
      }}
      config={baseConfig as any}
      useResizeHandler
      style={{ width: '100%' }}
    />
  )
}
