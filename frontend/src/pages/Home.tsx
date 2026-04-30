import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  fetchAggByTemp,
  fetchCurves,
  fetchSummary,
  paperUrl,
} from '../api'
import type {
  AggByTempResponse,
  CurvesResponse,
  DatasetSummary,
} from '../types'
import MetricGrid from '../components/MetricGrid'
import Plot, { baseConfig, usePlotLayout } from '../components/Plot'
import { useT } from '../i18n'

function fmtFreq(hz: number): string {
  if (hz >= 1000) return `${(hz / 1000).toFixed(1).replace(/\.0$/, '')} kHz`
  if (hz >= 1) return `${hz.toFixed(1).replace(/\.0$/, '')} Hz`
  return `${(hz * 1000).toFixed(0)} mHz`
}

export default function Home() {
  const t = useT()
  const baseLayout = usePlotLayout()
  const [summary, setSummary] = useState<DatasetSummary | null>(null)
  const [agg, setAgg] = useState<AggByTempResponse | null>(null)
  const [aging, setAging] = useState(0)
  const [curves, setCurves] = useState<CurvesResponse | null>(null)
  const [tab, setTab] = useState<'scatter' | 'temp'>('scatter')

  useEffect(() => {
    fetchSummary().then((s) => {
      setSummary(s)
      setAging(s.agings[0])
    })
    fetchAggByTemp().then(setAgg)
  }, [])

  useEffect(() => {
    if (summary) fetchCurves(aging).then(setCurves)
  }, [aging, summary])

  if (!summary) return <div className="banner">{t.common.loading}</div>

  return (
    <>
      <h1 className="page-title">{t.home.title}</h1>
      <div className="page-caption">{t.home.caption}</div>
      <div className="hero-meta">
        <span><strong>{t.common.semesterProject}</strong> · Davide Corso & Marco Soldani</span>
        <span className="dot">•</span>
        <span>{t.home.polimi} <strong>Politecnico di Milano</strong></span>
        <span className="dot">•</span>
        <a href={paperUrl} target="_blank" rel="noreferrer">{t.home.downloadPaper}</a>
      </div>

      <p>{t.home.intro}</p>

      <div style={{ marginTop: '0.8rem' }}>
        <MetricGrid
          items={[
            [t.home.measurements, summary.rows],
            [t.home.combos, summary.n_combinations],
            [t.home.curves, summary.n_curves],
            [t.home.freqRange, `${fmtFreq(summary.freq_min)} – ${fmtFreq(summary.freq_max)}`],
          ]}
        />
      </div>

      <hr />

      <div className="cols-2">
        <div>
          <h3>{t.home.structureTitle}</h3>
          <table>
            <thead>
              <tr><th>{t.home.dimension}</th><th>{t.home.levels}</th></tr>
            </thead>
            <tbody>
              <tr><td>Aging</td><td>{summary.agings.join(', ')}</td></tr>
              <tr><td>{t.home.tempAxis}</td><td>{summary.temperatures.join(', ')}</td></tr>
              <tr><td>SOC</td><td>{summary.socs.join(', ')}</td></tr>
            </tbody>
          </table>
          <p style={{ marginTop: '0.6rem' }}>{t.home.structureNote}</p>
        </div>

        <div>
          <h3>{t.home.jumpTitle}</h3>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            <Link to="/task/1" className="card" style={{ display: 'block' }}>
              <strong>{t.nav.task1}</strong>
              <div style={{ color: 'var(--text-soft)', fontSize: '0.88rem' }}>{t.home.task1Card}</div>
            </Link>
            <Link to="/task/2" className="card" style={{ display: 'block' }}>
              <strong>{t.nav.task2}</strong>
              <div style={{ color: 'var(--text-soft)', fontSize: '0.88rem' }}>{t.home.task2Card}</div>
              <div style={{ color: 'var(--text-faint)', fontSize: '0.78rem' }} />
            </Link>
            <Link to="/task/3" className="card" style={{ display: 'block' }}>
              <strong>{t.nav.task3}</strong>
              <div style={{ color: 'var(--text-soft)', fontSize: '0.88rem' }}>{t.home.task3Card}</div>
            </Link>
            <Link to="/docs" className="card" style={{ display: 'block' }}>
              <strong>{t.nav.docs}</strong>
              <div style={{ color: 'var(--text-soft)', fontSize: '0.88rem' }}>{t.home.docsCard}</div>
            </Link>
          </div>
        </div>
      </div>

      <hr />

      <h3>{t.home.glanceTitle}</h3>
      <div className="tabs">
        <button className={tab === 'scatter' ? 'active' : ''} onClick={() => setTab('scatter')}>{t.home.tabScatter}</button>
        <button className={tab === 'temp' ? 'active' : ''} onClick={() => setTab('temp')}>{t.home.tabTemp}</button>
      </div>

      {tab === 'scatter' && (
        <div className="card">
          <label className="field" style={{ maxWidth: 260 }}>
            {t.home.agingLevel}
            <select value={aging} onChange={(e) => setAging(Number(e.target.value))}>
              {summary.agings.map((a) => (<option key={a} value={a}>{a}</option>))}
            </select>
          </label>
          {curves && (
            <Plot
              data={curves.series.map((s) => ({
                type: 'scatter', mode: 'markers',
                x: s.z_real, y: s.z_imag_neg,
                marker: { size: 6, color: s.color },
                name: `${s.temperature}°C`,
                hovertemplate: 'Re(Z)=%{x:.3f} mΩ<br>-Im(Z)=%{y:.3f} mΩ<br>' + `T=${s.temperature}°C<extra></extra>`,
              }))}
              layout={{
                ...baseLayout,
                title: `Aging ${aging} — ${t.home.nyquistTitle}`,
                xaxis: { title: 'Re(Z) [mΩ]' },
                yaxis: { title: '-Im(Z) [mΩ]', scaleanchor: 'x', scaleratio: 1 },
                height: 500,
                legend: { orientation: 'h', y: -0.2 },
              }}
              config={baseConfig as any}
              useResizeHandler
              style={{ width: '100%' }}
            />
          )}
        </div>
      )}

      {tab === 'temp' && agg && (
        <div className="card">
          <Plot
            data={agg.series.map((s) => ({
              type: 'scatter', mode: 'lines+markers',
              x: s.temperature, y: s.z_real_mean,
              name: s.label,
              line: { color: s.color, width: 2 },
              marker: { size: 7, color: s.color },
            }))}
            layout={{
              ...baseLayout,
              title: t.home.tempPlotTitle,
              xaxis: { title: t.home.tempAxis },
              yaxis: { title: t.home.meanReZ },
              height: 420,
            }}
            config={baseConfig as any}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </div>
      )}
    </>
  )
}
