// One status chip for the whole site: case ladder rungs, outcome verdicts and ties. Colour is reserved for status:
// green proven or verified, amber partial or tie, red failed, blue predicted, a filled green for a public tool.
const KIND: Record<string, string> = {
  solved: "tool", "improves a public tool": "tool", confirmed: "ok", predicted: "pred", proven: "ok", verified: "ok",
  accepted: "ok", "verified equation": "ok", partial: "tie", tie: "tie", ambiguous: "tie", "needs one more experiment": "tie",
  "asks for an experiment": "tie", "not yet validated": "neutral", open: "neutral", "no clean equation": "bad", no_fit: "bad",
  failed: "bad", miss: "bad", hit: "ok", tied: "tie", rejected: "bad", approximation: "neutral", untestable: "neutral",
  proposed: "neutral",
};

export function StatusChip({ status, label }: { status: string; label?: string }) {
  const kind = KIND[status.toLowerCase()] ?? "neutral";
  const text = label ?? status.charAt(0).toUpperCase() + status.slice(1);
  return <span className={`status-chip status-${kind}`}>{text}</span>;
}
