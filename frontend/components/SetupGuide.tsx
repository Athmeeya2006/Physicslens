"use client";

import { useState } from "react";
import { PencilRuler } from "lucide-react";
import { isGallery, requestSetupGuide, type SetupGuideView } from "@/lib/api";

interface Props {
  runId: string;
  stepRef: number;
  instruction: string;
}

const CAPTION = "Illustration by Muse Image. A drawing of the setup, not a measurement.";

// The written instruction comes from the simulated experiment design and always leads; the picture only illustrates it.
export function SetupGuide({ runId, stepRef, instruction }: Props) {
  const [guide, setGuide] = useState<SetupGuideView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const show = async () => {
    setBusy(true);
    setError(null);
    try {
      setGuide(await requestSetupGuide(runId, stepRef));
    } catch (e) {
      setError(isGallery(runId) ? "No picture guide was stored for this example." : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const picture = guide?.status === "picture" && guide.image_url;
  return (
    <div className="setup-guide">
      <p className="setup-instruction">{guide?.instruction ?? instruction}</p>
      {picture ? (
        <figure>
          <div className="setup-pair">
            {guide.frame_url && <img src={guide.frame_url} alt="The frame from your clip, unchanged" />}
            <img src={guide.image_url} alt={`The setup drawn on your frame: ${guide.instruction}`} />
          </div>
          <figcaption className="small muted">{guide.caption || CAPTION}</figcaption>
        </figure>
      ) : (
        <>
          {guide?.reason && <p className="small muted">{guide.reason}</p>}
          {!guide && (
            <button type="button" className="secondary" onClick={show} disabled={busy}>
              <PencilRuler size={14} strokeWidth={2.2} aria-hidden="true" /> {busy ? "Drawing the setup..." : "Show me"}
            </button>
          )}
        </>
      )}
      {error && <p className="small error">{error}</p>}
    </div>
  );
}
