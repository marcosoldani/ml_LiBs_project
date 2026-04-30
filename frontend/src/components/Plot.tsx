import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import type { ComponentType } from 'react'
import { useTheme } from '../i18n'

interface PlotProps {
  data: any[]
  layout?: any
  config?: any
  style?: React.CSSProperties
  useResizeHandler?: boolean
  className?: string
  onClick?: any
  onHover?: any
}

const RawPlot = createPlotlyComponent(Plotly as any) as unknown as ComponentType<PlotProps>

export default RawPlot

const lightLayout: any = {
  template: 'plotly_white',
  margin: { l: 50, r: 20, t: 50, b: 50 },
  font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#0b1320' },
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  xaxis: { gridcolor: '#e3e8f0', zerolinecolor: '#cbd5e1', linecolor: '#cbd5e1' },
  yaxis: { gridcolor: '#e3e8f0', zerolinecolor: '#cbd5e1', linecolor: '#cbd5e1' },
}

const darkLayout: any = {
  template: 'plotly_dark',
  margin: { l: 50, r: 20, t: 50, b: 50 },
  font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#e6ecf6' },
  paper_bgcolor: '#131c30',
  plot_bgcolor: '#131c30',
  xaxis: { gridcolor: '#1e2a44', zerolinecolor: '#2a3a5c', linecolor: '#2a3a5c' },
  yaxis: { gridcolor: '#1e2a44', zerolinecolor: '#2a3a5c', linecolor: '#2a3a5c' },
}

export const baseConfig: any = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d'],
}

// Backwards-compatible export — pages currently spread `baseLayout`.
// We expose a hook that returns the right one for the current theme.
export const baseLayout: any = lightLayout

export function usePlotLayout(): any {
  const { theme } = useTheme()
  return theme === 'dark' ? darkLayout : lightLayout
}
