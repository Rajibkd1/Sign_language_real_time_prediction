import { NavLink, Route, Switch } from 'react-router-dom';
import { BrainCircuit, Camera, FileVideo, History, Home, Info, Menu, Moon, Sun, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import AmbientBackground from './components/AmbientBackground.jsx';
import AppFooter from './components/AppFooter.jsx';
import ToastHost, { showToast } from './components/ToastHost.jsx';
import RecognitionPage from './pages/RecognitionPage.jsx';
import SplashPage from './pages/SplashPage.jsx';
import VideoUploadPage from './pages/VideoUploadPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';
import ModelPage from './pages/ModelPage.jsx';
import AboutPage from './pages/AboutPage.jsx';

const navItems = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/live', label: 'Live Camera', icon: Camera },
  { to: '/upload', label: 'Video Upload', icon: FileVideo },
  { to: '/history', label: 'History', icon: History },
  { to: '/model', label: 'Model', icon: BrainCircuit },
  { to: '/about', label: 'About', icon: Info },
];

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('bdsl-theme') || 'dark');
  const [navOpen, setNavOpen] = useState(false);
  const [compactHeader, setCompactHeader] = useState(false);
  const drawerTouchStartX = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('bdsl-theme', theme);
  }, [theme]);

  useEffect(() => {
    const onScroll = () => setCompactHeader(window.scrollY > 28);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('nav-lock', navOpen);
    return () => document.body.classList.remove('nav-lock');
  }, [navOpen]);

  useEffect(() => {
    const media = window.matchMedia('(min-width: 901px)');
    const closeOnDesktop = () => {
      if (media.matches) setNavOpen(false);
    };
    closeOnDesktop();
    media.addEventListener?.('change', closeOnDesktop);
    return () => media.removeEventListener?.('change', closeOnDesktop);
  }, []);

  const closeNav = () => setNavOpen(false);
  const navClass = compactHeader ? 'top-nav is-compact' : 'top-nav';

  const toggleTheme = () => {
    setTheme((value) => {
      const next = value === 'dark' ? 'light' : 'dark';
      showToast({ type: 'success', message: `${next === 'dark' ? 'Dark' : 'Light'} theme enabled` });
      return next;
    });
  };

  const handleDrawerTouchStart = (event) => {
    drawerTouchStartX.current = event.touches?.[0]?.clientX || 0;
  };

  const handleDrawerTouchEnd = (event) => {
    const endX = event.changedTouches?.[0]?.clientX || drawerTouchStartX.current;
    if (endX - drawerTouchStartX.current > 60) {
      closeNav();
    }
  };

  return (
    <div className="app-shell">
      <AmbientBackground />
      <header className={navClass}>
        <NavLink to="/" className="brand" aria-label="BdSL Vision recognition" onClick={closeNav}>
          <span className="brand-mark" aria-hidden="true">
            <BrainCircuit size={18} />
          </span>
          <span className="brand-text">BdSL Vision</span>
        </NavLink>

        <nav className="nav-links desktop-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink key={item.to} to={item.to} exact={item.to === '/'} className="nav-link" activeClassName="active" data-tooltip={item.label}>
                <Icon size={15} aria-hidden="true" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <div className="nav-actions">
          <button
            type="button"
            className="button ghost icon-button menu-button"
            onClick={() => setNavOpen(true)}
            aria-label="Open navigation menu"
            aria-expanded={navOpen}
            aria-controls="mobile-navigation"
            data-tooltip="Menu"
          >
            <Menu size={20} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="button ghost icon-button"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            data-tooltip={theme === 'dark' ? 'Light theme' : 'Dark theme'}
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </div>
      </header>

      <button
        type="button"
        className={`nav-backdrop${navOpen ? ' open' : ''}`}
        aria-label="Close navigation menu"
        onClick={closeNav}
        tabIndex={navOpen ? 0 : -1}
      />
      <aside
        id="mobile-navigation"
        className={`nav-drawer${navOpen ? ' open' : ''}`}
        aria-hidden={!navOpen}
        onTouchStart={handleDrawerTouchStart}
        onTouchEnd={handleDrawerTouchEnd}
      >
        <div className="drawer-topline">
          <span>Navigation</span>
          <button type="button" className="button ghost icon-button" onClick={closeNav} aria-label="Close navigation menu" data-tooltip="Close menu">
            <X size={20} aria-hidden="true" />
          </button>
        </div>
        <nav className="drawer-nav" aria-label="Mobile navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                exact={item.to === '/'}
                className="drawer-nav-link"
                activeClassName="active"
                onClick={closeNav}
              >
                <Icon size={18} aria-hidden="true" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>
      </aside>

      <Switch>
        <Route exact path="/">
          <SplashPage />
        </Route>
        <Route path="/live">
          <RecognitionPage />
        </Route>
        <Route path="/upload">
          <VideoUploadPage />
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
      <AppFooter />
      <ToastHost />
    </div>
  );
}
