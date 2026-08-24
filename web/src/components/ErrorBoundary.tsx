import { Component, type ReactNode } from "react";

interface ErrorBoundaryState {
  failed: boolean;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch() {
    // Deliberately omit arbitrary error objects from the browser console.
  }

  render() {
    if (this.state.failed) {
      return (
        <main className="fatal-state">
          <div className="fatal-state__mark" aria-hidden="true">
            !
          </div>
          <p className="eyebrow">Interface fault</p>
          <h1>The inspection console could not be rendered.</h1>
          <p>Reload the page. If the issue persists, contact the service operator.</p>
          <button type="button" onClick={() => window.location.reload()}>
            Reload console
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}
