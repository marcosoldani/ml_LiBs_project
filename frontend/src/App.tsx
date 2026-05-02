import { useEffect, useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Task1 from './pages/Task1'
import Task2 from './pages/Task2'
import Task3 from './pages/Task3'
import Documentation from './pages/Documentation'
import { LangContext, ThemeContext, type Lang, type Theme } from './i18n'

function loadInitial<T extends string>(key: string, fallback: T, valid: T[]): T {
  if (typeof window === 'undefined') return fallback
  const v = window.localStorage.getItem(key) as T | null
  return v && valid.includes(v) ? v : fallback
}

export default function App() {
  const [lang, setLang] = useState<Lang>(() =>
    loadInitial<Lang>('battery-mlops-lang', 'it', ['it', 'en']),
  )
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = loadInitial<Theme | ''>('battery-mlops-theme', '', ['light', 'dark'])
    if (saved === 'light' || saved === 'dark') return saved
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
      return 'dark'
    }
    return 'light'
  })

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('battery-mlops-lang', lang)
    document.documentElement.lang = lang
  }, [lang])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('battery-mlops-theme', theme)
    document.documentElement.dataset.theme = theme
  }, [theme])

  return (
    <LangContext.Provider value={{ lang, setLang }}>
      <ThemeContext.Provider value={{ theme, setTheme }}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Home />} />
            <Route path="/task/1" element={<Task1 />} />
            <Route path="/task/2" element={<Task2 />} />
            <Route path="/task/3" element={<Task3 />} />
            <Route path="/docs" element={<Documentation />} />
            <Route path="*" element={<Home />} />
          </Route>
        </Routes>
      </ThemeContext.Provider>
    </LangContext.Provider>
  )
}
