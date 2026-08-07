import { NavLink, Route, Switch } from 'react-router-dom';
import { BrainCircuit, Camera, History, Info, Moon, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import RecognitionPage from './pages/RecognitionPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';
import ModelPage from './pages/ModelPage.jsx';
import AboutPage from './pages/AboutPage.jsx';

const navItems = [
  { to: '/', label: 'Recognition', icon: Camera },
  { to: '/history', label: 'History', icon: History },
  { to: '/model', label: 'Model', icon: BrainCircuit },
  { to: '/about', label: 'About', icon: Info }
];

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('bdsl-theme') || 'dark');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('bdsl-theme', theme);
  }, [theme]);

  return (
    <div className="app-shell">
      <header className="top-nav">
        <NavLink to="/" className="brand" aria-label="BdSL Vision recognition">
          <span className="brand-mark" aria-hidden="true">
            <BrainCircuit size={20} />
          </span>
          <span>BdSL Vision</span>
        </NavLink>

        <nav className="nav-links" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink key={item.to} to={item.to} exact={item.to === '/'} className="nav-link" activeClassName="active">
                <Icon size={16} aria-hidden="true" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <div className="nav-actions">
          <button
            type="button"
            className="button ghost icon-button"
            onClick={() => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      <Switch>
        <Route exact path="/">
          <RecognitionPage />
        </Route>
        <Route path="/history">
          <HistoryPage />
        </Route>
        <Route path="/model">
          <ModelPage />
        </Route>
        <Route path="/about">
          <AboutPage />
        </Route>
      </Switch>
    </div>
  );
}
