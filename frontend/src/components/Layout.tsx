import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { fetchProject } from '../api'
import type { ProjectInfo } from '../types'
import { useLang, useT, useTheme } from '../i18n'

export default function Layout() {
  const [project, setProject] = useState<ProjectInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { lang, setLang } = useLang()
  const { theme, setTheme } = useTheme()
  const t = useT()

  useEffect(() => {
    fetchProject()
      .then(setProject)
      .catch(() => setError(t.common.apiError))
  }, [t])

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <div className="app-brand">
            <span>Battery GEIS MLOps</span>
          </div>
          <nav className="app-nav">
            <NavLink to="/" end>{t.nav.home}</NavLink>
            <NavLink to="/task/1">{t.nav.task1}</NavLink>
            <NavLink to="/task/2">{t.nav.task2}</NavLink>
            <NavLink to="/task/3">{t.nav.task3}</NavLink>
            <NavLink to="/docs">{t.nav.docs}</NavLink>
          </nav>
          <div className="app-header-spacer" />
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={theme === 'dark' ? 'Light mode' : 'Dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <div className="lang-switch">
            <button
              type="button"
              className={lang === 'it' ? 'active' : ''}
              onClick={() => setLang('it')}
              aria-label="Italiano"
              title="Italiano"
            >
              IT
            </button>
            <button
              type="button"
              className={lang === 'en' ? 'active' : ''}
              onClick={() => setLang('en')}
              aria-label="English"
              title="English"
            >
              EN
            </button>
          </div>
          {project && (
            <div className="app-header-meta">
              v{project.version} · THERMINIC 2025
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        {error && <div className="banner banner-error">{error}</div>}
        <Outlet context={{ project }} />
      </main>

      <footer className="app-footer">
        {project ? (
          <>
            <strong>{project.name}</strong>
            <span className="dot">•</span> v{project.version}
            <span className="dot">•</span> {t.footer.authors}: {project.authors.join(' · ')}
            <br />
            <span>{t.footer.line}</span>
          </>
        ) : (
          <span>{t.common.semesterProject}</span>
        )}
      </footer>
    </div>
  )
}
