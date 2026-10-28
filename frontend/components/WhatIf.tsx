"use client";

import { useRef, useState } from "react";
import type { TrackPoint } from "@/lib/api";
import { accessCode, API_BASE, askWhatIf } from "@/lib/api";
import { startMic, type Mic } from "@/lib/voice";

export interface WhatIfAnswer {
  question: string;
  understood: boolean;
  description?: string;
  reply: string;
  points?: TrackPoint[];
  period_s?: number;
  length_m?: number;
  model?: string;
}

interface Props {
  runId: string;
  onGhost: (ghost: { label: string; points: TrackPoint[] } | null) => void;
}

export function WhatIf({ runId, onGhost }: Props) {
  const [question, setQuestion] = useState("");
  const [answers, setAnswers] = useState<WhatIfAnswer[]>([]);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mic = useRef<Mic | null>(null);
  const socket = useRef<WebSocket | null>(null);

  const show = (a: WhatIfAnswer) => {
    setAnswers((prev) => [a, ...prev].slice(0, 5));
    onGhost(a.understood && a.points ? { label: a.description ?? a.question, points: a.points } : null);
  };

  const ask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setError(null);
    try {
      show(await askWhatIf(runId, question.trim()));
      setQuestion("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stopListening = () => {
    mic.current?.stop();
    mic.current = null;
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ type: "stop" }));
    setListening(false);
  };

  const listen = async () => {
    setError(null);
    setTranscript("");
    const ws = new WebSocket(API_BASE.replace(/^http/, "ws") + "/ws/voice");
    socket.current = ws;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "transcript") setTranscript(msg.text);
      if (msg.type === "whatif") show(msg as WhatIfAnswer);
      if (msg.type === "notice") setError(msg.message);
    };
    ws.onclose = () => stopListening();
    ws.onopen = async () => {
      ws.send(JSON.stringify({ run_id: runId, access_code: accessCode() }));
      try {
        mic.current = await startMic((pcm) => ws.readyState === WebSocket.OPEN && ws.send(pcm));
        setListening(true);
      } catch {
        setError("Microphone unavailable (on a phone this page must be opened over HTTPS).");
        ws.close();
      }
    };
  };

  return (
    <div className="panel">
      <h3>What if?</h3>
      <p className="muted small" style={{ marginBottom: 12 }}>Ask out loud or type, for example &ldquo;what if the string were twice as long?&rdquo; The language model only turns the question into a parameter change; the fitted equation is re-simulated and drawn as a pink ghost.</p>
      <div className="row">
        <button type="button" onClick={listening ? stopListening : listen} className={listening ? "listening" : ""}>
          {listening ? "Stop listening" : "Ask by voice"}
        </button>
        <form onSubmit={ask} className="whatif-form">
          <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Type a what-if question" />
          <button type="submit" disabled={busy || !question.trim()}>{busy ? "..." : "Ask"}</button>
        </form>
      </div>
      {listening && <p className="transcript">{transcript || "Listening..."}</p>}
      {error && <p className="error small">{error}</p>}
      {answers.map((a, i) => (
        <div key={i} className="whatif-answer">
          <p><strong>{a.understood ? a.description : "Not a change I can simulate"}</strong>: {a.reply}</p>
          {a.period_s && <p className="muted small">New period about {a.period_s.toFixed(2)} s{a.length_m ? `, length ${a.length_m.toFixed(2)} m` : ""}.</p>}
        </div>
      ))}
    </div>
  );
}
