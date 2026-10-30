// Errors of the demo carry a code, so a caller decides what to do from the code and not from the message text.
// A plain Error (a broken invariant, or the browser's own error) has none: where codes are told apart it is given
// one (withCode: a city load, a device rebuild), and a computation's failure card follows the stage it stopped at
// (app/pipeline.js), whatever the error.

/** The kinds of failure a caller tells apart. */
export const ErrorCode = Object.freeze({
  TIMEOUT: "timeout",          // a step did not finish in its time limit (withLimit)
  CANCELLED: "cancelled",      // superseded by a newer request (another city, a newer computation)
  DOWNLOAD: "download",        // a file could not be downloaded
  GPU: "gpu",                  // no usable WebGPU device, or the device failed
  PLANNER: "planner",          // the route planner threads did not start or failed
  MODEL: "model",              // the model files do not fit this page (a reload does not help)
  MISSION: "mission",          // no random mission could be drawn in this city
  TOO_LARGE: "too-large",      // a city needs more graphics memory than the demo uses (gpu/predictor.js budget)
});

/** An error with an ErrorCode. */
export class DemoError extends Error {
  /** @param {string} code one of ErrorCode  @param {string} message shown in the console and failure texts */
  constructor(code, message) {
    super(message);
    this.name = "DemoError";
    this.code = code;
  }
}

/** `error` itself when it has a code, else a DemoError `code` with its message (the browser's own errors). */
export const withCode = (error, code) => (error instanceof DemoError ? error : new DemoError(code, error.message));

/** Whether `error` is a DemoError with this code. */
export const hasCode = (error, code) => error instanceof DemoError && error.code === code;

/** An error as text for the console or a worker's reply: its stack (which starts with its message), else its text. */
export const errorText = (error) => error?.stack || error?.message || String(error);

/** The model's files do not fit this page (DemoError MODEL: a reload does not help). */
export const modelMisfit = (message) => new DemoError(ErrorCode.MODEL, message);
