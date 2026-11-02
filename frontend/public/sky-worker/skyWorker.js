// Runs the lab's integrator off the page's thread: the newest request is answered, older ones are dropped by the page.
import { integrate } from "./skyIntegrate.js"; // .js: the compiled worker is loaded by the browser as is
self.onmessage = (e) => {
    const { id, init, changes, target, base } = e.data;
    try {
        self.postMessage({ id, bodies: integrate(init, changes, target, 1500, base) });
    }
    catch (err) {
        self.postMessage({ id, error: err.message });
    }
};
