// The shared-memory protocol between a planner thread and its helper threads (helper_pool.js, helper_worker.js,
// trajopt.js): the tasks, the slots of the control array, and the per-node sizes of the shared outputs.

/** The helpers' tasks (ctrl[CTRL.TASK]): wind grid rows, lattice node winds, optimizer node terms / node values. */
export const TASK = Object.freeze({ GRID: 1, LATTICE: 2, NODE_TERMS: 3, NODE_VALUES: 4 });
/**
 * Slots of the control array (Int32): generation, helpers done, node count, participants, error flag, task, and
 * from READY one ready flag per helper.
 */
export const CTRL = Object.freeze({ GENERATION: 0, DONE: 1, N: 2, PARTICIPANTS: 3, ERROR: 4, TASK: 5, READY: 6 });
/** Doubles of a node's state y = (x, v, a) (trajopt.js). */
export const NODE_STATE = 9;
/** Node k's state in a buffer of node states ([node][NODE_STATE]). */
export const nodeStateOf = (buffer, k) => buffer.subarray(NODE_STATE * k, NODE_STATE * (k + 1));
/**
 * Where a node's terms start in its NODE_TERM doubles: the gradient at 0 (NODE_STATE), the Hessian (NODE_STATE^2),
 * the dt-cross terms (NODE_STATE), the dt term (1).
 */
export const TERM_HESSIAN = NODE_STATE, TERM_DT_CROSS = TERM_HESSIAN + NODE_STATE * NODE_STATE;
export const TERM_DT = TERM_DT_CROSS + NODE_STATE;
/**
 * A node's margins at most (trajopt.js nodeEval): its own 4 (clearance, acceleration, airspeed, thrust) and up to
 * MAX_EXTRA_MARGINS from the fields' extra ones (the corridor's 6 edges, the flight band and the planning domain: 12).
 */
export const MAX_EXTRA_MARGINS = 20, MAX_MARGINS = 4 + MAX_EXTRA_MARGINS;
/**
 * Doubles per node in the shared outputs: node terms (above), and node values (trajopt.js nodeValueTerms: their
 * count, the energy, a barrier per margin, the 3 tracking terms).
 */
export const NODE_TERM = TERM_DT + 1, VALUE_TERMS = 1 + 1 + MAX_MARGINS + 3;
/** Participant p's share [lo, hi) of n items split among `parts` (0 = the planner thread itself). */
export const shareOf = (p, parts, n) => [Math.floor(p * n / parts), Math.floor((p + 1) * n / parts)];
