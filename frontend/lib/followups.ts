// The agent's experiment requests that were answered by a later recording: the original run, the run that answers it,
// and one line on what the answer settled (from the answering result). A request is listed here only when the second
// recording really is the experiment the agent designed (same setup, the change it asked for).
// headline: what the answering recording settled, in one sentence; its page leads with it, and the verdict generated
// for that recording (which may name a new, finer tie) follows as "still open".
export interface Followup { answer: string; question: string; settled: string; chip: string; headline: string }

export const FOLLOWUPS: Record<string, Followup> = {
  "gallery-pendulum": {
    answer: "gallery-pendulum-longer", question: "gallery-pendulum", chip: "Answered: damping confirmed",
    headline: "The question the agent asked is settled: damping is real. On the longer recording the simple pendulum falls 121 BIC behind.",
    settled: "The same swing filmed for 12.6 s instead of 6.5 settles the question it asked: the simple pendulum falls 121 BIC behind and damping forecasts the unseen motion within 1.9%. A finer tie remains (linear damping or air drag), and the agent designs the recording for it.",
  },
  "gallery-spring": {
    answer: "gallery-spring-longer", question: "gallery-spring", chip: "Answered: dry friction confirmed",
    headline: "The question the agent asked is settled: the glider slows by dry friction, not viscous damping.",
    settled: "The same pull filmed for 20 s settles it: dry friction forecasts the unseen motion within 3.4%; viscous damping misses by 18.7% and falls 12 BIC behind.",
  },
};

// The request a result answers, if it answers one.
export const answering = (runId: string): Followup | undefined => Object.values(FOLLOWUPS).find((f) => f.answer === runId);
