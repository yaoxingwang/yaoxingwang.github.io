// Downloads with an end: the HTTP status is checked, and a download that receives nothing for STALL_MS is given up,
// so a stalled or missing file never leaves the page waiting for ever or reports an unreadable file.
import { DemoError, ErrorCode } from "./errors.js";

const STALL_MS = 60000;

/**
 * The bytes of `url` (Uint8Array).  A bad status, a stall or no connection throws DemoError DOWNLOAD with a readable
 * message.  `onBytes(n)` reports the bytes received so far.
 */
export async function download(url, onBytes = null) {
  const controller = new AbortController();     // aborted by a stall
  let timer = null;
  const arm = () => { clearTimeout(timer); timer = setTimeout(() => controller.abort(), STALL_MS); };
  const name = url.split("/").pop();
  try {
    arm();
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new DemoError(ErrorCode.DOWNLOAD, `${name} could not be downloaded (server answered ${response.status})`);
    }
    const chunks = [];
    let size = 0;
    for (const reader = response.body.getReader(); ;) {
      arm();
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      size += value.length;
      onBytes?.(size);
    }
    const bytes = new Uint8Array(size);
    let at = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, at);
      at += chunk.length;
    }
    return bytes;
  } catch (error) {
    if (controller.signal.aborted) {
      throw new DemoError(ErrorCode.DOWNLOAD, `${name}: nothing received for ${STALL_MS / 1000} s`);
    }
    if (error instanceof TypeError) {
      throw new DemoError(ErrorCode.DOWNLOAD, `${name} could not be downloaded (no connection to the server)`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/** The JSON document at `url` (download's errors). */
export const downloadJson = async (url) => JSON.parse(new TextDecoder().decode(await download(url)));
