// Microphone -> 24 kHz signed 16-bit little-endian mono PCM chunks (what the transcription stream expects).

const TARGET_RATE = 24000;
const CHUNK_SAMPLES = 2400; // 100 ms

// Runs on the audio thread: averages each output sample's window of input (a box filter, so higher
// frequencies don't fold back into speech), converts to Int16, posts ~100 ms chunks.
const WORKLET = `
class PcmChunker extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.ratio = sampleRate / options.processorOptions.target;
    this.acc = 0;
    this.sum = 0;
    this.count = 0;
    this.buf = new Int16Array(options.processorOptions.chunk);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      this.sum += ch[i];
      this.count += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const s = Math.max(-1, Math.min(1, this.sum / this.count));
        this.sum = 0;
        this.count = 0;
        this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.buffer.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-chunker", PcmChunker);
`;

export interface Mic {
  stop: () => void;
}

export async function startMic(onChunk: (pcm: ArrayBuffer) => void): Promise<Mic> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  // Ask for 24 kHz directly. Firefox refuses to connect a microphone to a context at a different rate than
  // the device, so then fall back to the device rate and let the worklet downsample.
  let ctx = new AudioContext({ sampleRate: TARGET_RATE });
  let source: MediaStreamAudioSourceNode;
  try {
    source = ctx.createMediaStreamSource(stream);
  } catch {
    await ctx.close();
    ctx = new AudioContext();
    source = ctx.createMediaStreamSource(stream);
  }
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const node = new AudioWorkletNode(ctx, "pcm-chunker", {
    processorOptions: { target: TARGET_RATE, chunk: CHUNK_SAMPLES },
  });
  node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => onChunk(e.data);
  source.connect(node);
  return {
    stop: () => {
      node.port.onmessage = null;
      source.disconnect();
      node.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
