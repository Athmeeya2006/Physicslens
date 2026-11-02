// Runs the lab's integrator off the page's thread: the newest request is answered, older ones are dropped by the page.

import type { Changes, Initial, SkyBody } from "./skyTypes.js";
import { integrate } from "./skyIntegrate.js";  // .js: the compiled worker is loaded by the browser as is

interface Job { id: number; init: Initial; changes: Changes; target: string; base: SkyBody[] }

self.onmessage = (e: MessageEvent<Job>) => {
  const { id, init, changes, target, base } = e.data;
  try {
    (self as unknown as Worker).postMessage({ id, bodies: integrate(init, changes, target, 1500, base) });
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: (err as Error).message });
  }
};
