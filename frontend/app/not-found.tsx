export default function NotFound() {
  return (
    <main className="stack" style={{ gap: 16 }}>
      <header className="page-head">
        <h1>Page not found</h1>
        <p className="tagline">There is nothing at this address. It may have been a mistyped link, or a result that was never saved.</p>
      </header>
      <section className="panel stack" style={{ gap: 10 }}>
        <p>Where to go from here:</p>
        <ul className="stack" style={{ gap: 6, paddingLeft: 18, margin: 0 }}>
          <li><a href="/">Analyze: real recordings, each a complete analysis with its decision trail</a></li>
          <li><a href="/study">Studies: goals planned and answered over several recordings</a></li>
          <li><a href="/dataset">Open dataset: every verified physical parameter</a></li>
        </ul>
      </section>
    </main>
  );
}
