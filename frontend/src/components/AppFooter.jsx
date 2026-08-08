import { BrainCircuit } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function AppFooter() {
  return (
    <footer className="app-footer">
      <div className="footer-brand">
        <span className="footer-mark" aria-hidden="true">
          <BrainCircuit size={15} />
        </span>
        <span>&copy; 2026 BdSL Vision</span>
      </div>
      <nav className="footer-links" aria-label="Footer navigation">
        <Link to="/model">Model 6 - Updated Aug 2026</Link>
        <Link to="/about">Docs</Link>
      </nav>
    </footer>
  );
}
