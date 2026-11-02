import { GLOSSARY } from "@/lib/glossary";

// Every technical word the pages use, in plain English. The same text appears as a tooltip wherever the word does.
export default function GlossaryPage() {
  return (
    <main id="main">
      <header className="page-head">
        <h1>Glossary</h1>
        <p className="tagline">The technical words on these pages, in plain English.</p>
      </header>
      <dl className="glossary panel">
        {Object.entries(GLOSSARY).map(([k, g]) => (
          <div key={k} id={k}>
            <dt><strong>{g.term}</strong></dt>
            <dd>{g.text}</dd>
          </div>
        ))}
      </dl>
    </main>
  );
}
