// Downloads and decodes one CFD wind file off the page's main thread (cfd.js loadCfd).  Messages to the page: download
// progress { bytes }, decoding progress { decoded } (every DECODE_REPORT_SHARE), then { values } (shared memory when
// the page is cross-origin isolated, otherwise transferred) or { error, code }.
import { download } from "../shared/download.js";
import { decodeCfd } from "./cfd.js";

const DECODE_REPORT_SHARE = 0.02;   // decoding progress is reported in steps of this share

self.onmessage = async ({ data: { url, shape } }) => {
  try {
    const bytes = await download(url, (n) => self.postMessage({ bytes: n }));
    let reported = 0;
    const values = decodeCfd(bytes.buffer, shape, (share) => {
      if (share - reported < DECODE_REPORT_SHARE && share !== 1) return;
      reported = share;
      self.postMessage({ decoded: share });
    });
    self.postMessage({ values }, values.buffer instanceof ArrayBuffer ? [values.buffer] : []);
  } catch (error) {
    // only the page's own error codes (shared/errors.js) are passed on, not a browser error's numeric code
    // the message alone: it becomes the page's DemoError message (wind/cfd.js)
    self.postMessage({ error: error.message, code: typeof error.code === "string" ? error.code : undefined });
  }
};
