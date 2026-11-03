// The agent's experiment requests that were answered by a later recording: the original run, the run that answers it,
// and one line on what the answer settled (from the answering result). A request is listed here only when the second
// recording really is the experiment the agent designed (same setup, the change it asked for).
export interface Followup { answer: string; question: string; settled: string }

export const FOLLOWUPS: Record<string, Followup> = {
  "gallery-spring": {
    answer: "gallery-spring-longer", question: "gallery-spring",
    settled: "The same pull filmed for 20 s settles it: dry friction forecasts the unseen motion within 3.4%; viscous damping misses by 18.7% and falls 12 BIC behind.",
  },
};

// The request a result answers, if it answers one.
export const answering = (runId: string): Followup | undefined => Object.values(FOLLOWUPS).find((f) => f.answer === runId);
