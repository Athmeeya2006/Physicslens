"use client";

import { useEffect } from "react";

// Any page that fails to render shows this instead of an empty, dark screen: what happened, and two ways on.
export default function PageError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <main id="main">
      <section className="panel stack" role="alert">
        <h1>This part of the page could not be shown</h1>
        <p>Something in the data for this page was not what the page expected. Nothing was lost: try again, or go back to
          the home page.</p>
        <p className="small muted">Details: {error.message}</p>
        <div className="row" style={{ gap: 10 }}>
          <button type="button" onClick={reset}>Try again</button>
          <a className="button secondary" href="/">Home page</a>
        </div>
      </section>
    </main>
  );
}
