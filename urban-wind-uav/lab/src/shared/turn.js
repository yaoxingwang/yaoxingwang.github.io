// A turn for the page (or a worker) between steps of a long computation.

/** Work on the page's main thread for longer than this gives the page a turn (long read-backs, post-processing). */
export const TURN_AFTER_MS = 8;

const turnChannel = new MessageChannel();
const turnQueue = [];
turnChannel.port1.onmessage = () => turnQueue.shift()?.();

/**
 * A turn without setTimeout's minimum delay: a message to ourselves runs after pending events and frames (and is not
 * throttled in background tabs).
 */
export const yieldTurn = () => new Promise((resolve) => { turnQueue.push(resolve); turnChannel.port2.postMessage(0); });
