// A sticky row of jump links to every section of a long page: everything stays on the page, this only saves scrolling.
export function OnThisPage({ items }: { items: { id: string; label: string }[] }) {
  return (
    <nav className="on-page" aria-label="On this page">
      <span>On this page:</span>
      {items.map((i) => <a key={i.id} href={`#${i.id}`}>{i.label}</a>)}
    </nav>
  );
}
