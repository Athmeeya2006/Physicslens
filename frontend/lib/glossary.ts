// Plain-English meanings of the technical words the pages use, shown as tooltips (components/Term.tsx) and listed on
// /glossary. Summaries use plain words first; these are for the reader who wants the term.
export const GLOSSARY: Record<string, { term: string; text: string }> = {
  bic: { term: "BIC", text: "Bayesian information criterion: how well a law fits, with a penalty for every extra parameter. Lower is better; a gap above 10 is very strong evidence." },
  heldout: { term: "Held-out", text: "Data kept back while fitting. A law is judged on how well it forecasts this part, which it never saw." },
  chi2: { term: "Chi-square", text: "The squared misses divided by each measurement's own error, added up. About 1 per point means the misses are as big as the errors say they should be." },
  oc: { term: "O–C", text: "Observed minus calculated: how early or late each transit came compared with a constant-period prediction." },
  sigma: { term: "Sigma", text: "One standard deviation, the usual size of a measurement's error. Two sigma covers about 95% of cases." },
  ci95: { term: "95% interval", text: "The range that should contain the true value 95 times out of 100, if the error model is right." },
  bootstrap: { term: "Bootstrap", text: "Refitting many times on resampled data to see how much the answer moves: that spread is the uncertainty." },
  nrmse: { term: "Forecast error", text: "The forecast's typical miss as a share of the motion's own size (normalised RMS)." },
  ghost: { term: "Ghost", text: "The accepted law's simulation drawn over the real video, so you can see where it agrees and where it drifts." },
  sindy: { term: "PySINDy", text: "A model-free method that searches for a sparse set of terms that fit the data. Shown for comparison, never used to decide." },
};
