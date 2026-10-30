import { validateRoute } from './validate_route.js';

/** Reject routes before display/scoring, retaining evidence for inspection/export. */
export function validatePlans(plans, city, mission) {
  for (const plan of Object.values(plans)) {
    if (!plan.points) continue;
    plan.validation = validateRoute(plan.points, city, { mission, stepM: 0.25, minClearanceM: 10 });
    if (!plan.validation.pass) {
      plan.rejectedPoints = plan.points;
      plan.points = null;
      plan.failed = 'independent building-surface clearance check failed';
    }
  }
  return plans;
}
