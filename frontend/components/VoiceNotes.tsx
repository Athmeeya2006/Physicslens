"use client";

import { Mic, Square } from "lucide-react";
import { useRef, useState } from "react";
import { accessCode, API_BASE } from "@/lib/api";
import { startMic, type Mic as MicHandle } from "@/lib/voice";

export interface SpokenNotes {
  understood: boolean;
  heard: string;
  reply: string;
  pendulum_length_m?: number | null;
  object_size_m?: number | null;
  object_hint?: string | null;
  slow_motion?: number | null;
  label?: string | null;
}

/** Say the settings while filming ("the string is 50 centimeters, this is the left front shock"). */
export function VoiceNotes({ onNotes }: { onNotes: (n: SpokenNotes) => void }) {
  const [listening, setListening] = useState(false);
  const [text, setText] = useState("");
  const [reply, setReply] = useState<string | null>(null);
  const mic = useRef<MicHandle | null>(null);
  const socket = useRef<WebSocket | null>(null);

  const stop = () => {
    mic.current?.stop();
    mic.current = null;
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ type: "stop" }));
    setListening(false);
  };

  const start = () => {
    setText("");
    setReply(null);
    const ws = new WebSocket(API_BASE.replace(/^http/, "ws") + "/ws/voice/notes");
    socket.current = ws;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "transcript") setText(msg.text);
      if (msg.type === "notes") {
        setReply(msg.reply);
        onNotes(msg as SpokenNotes);
      }
      if (msg.type === "notice") setReply(msg.message);
    };
    ws.onclose = () => stop();
    ws.onopen = async () => {
      ws.send(JSON.stringify({ access_code: accessCode() }));
      try {
        mic.current = await startMic((pcm) => ws.readyState === WebSocket.OPEN && ws.send(pcm));
        setListening(true);
      } catch {
        setReply("Microphone unavailable (on a phone this page must be opened over HTTPS).");
        ws.close();
      }
    };
  };

  return (
    <div className="voice-notes">
      <button type="button" className={`secondary ${listening ? "listening" : ""}`} onClick={listening ? stop : start}>
        {listening ? <Square size={14} aria-hidden="true" /> : <Mic size={14} aria-hidden="true" />}
        {listening ? "Done" : "Say it"}
      </button>
      <span className="small muted">
        {listening ? text || "Listening: say the string length, the object's size, or which part this is." : reply ?? "Or say the settings out loud."}
      </span>
    </div>
  );
}
