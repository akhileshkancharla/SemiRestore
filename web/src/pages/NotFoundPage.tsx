import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <section className="not-found">
      <span className="not-found__code">404</span>
      <p className="eyebrow">Unknown console route</p>
      <h1>This inspection view does not exist.</h1>
      <Link className="button" to="/">
        Return to overview
      </Link>
    </section>
  );
}
