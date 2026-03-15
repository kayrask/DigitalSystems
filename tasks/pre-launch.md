# Pre-Launch Hardening — Post-Demo

> Audit completed: 2026-03-14. Do with Opus after dissertation demo.

## CRITICAL — Launch Blockers

1. JWT Auth Overhaul — every endpoint trusts user_id as a param without server-side validation
2. Fix IDOR — /scans/{id}, /explain, /simulate_outcome, /me PUT all expose other users data
3. File upload MIME validation — bytes saved before confirming its an image

## HIGH

4. CORS restrict allow_methods and allow_headers (currently wildcard)
5. CSRF protection on all POST/PUT
6. Frontend token expiry + auto-logout
7. Password reset / forgot-password flow
8. Pydantic Field length limits on name, address, allergies
9. More training data — bags (12 samples) and redness (12 samples) are too small for launch

## MEDIUM

10. Password minimum 8+ chars (currently 6)
11. Structured logging — replace print() with logging
12. Rate limit IP spoofing — configure trusted proxy headers
13. Admin promote endpoint — add rate limiting and audit log
14. DB connection TLS for remote MySQL

## ML Accuracy

15. Fix weak_positive 0.45 implicit floor (line 249 main.py) contradicts tuned thresholds
16. Validate YOLO fusion multiplier 0.50 (changed from 0.35 without A/B test)
17. Subject-level stratification in data splits

## Quick Polish (any time)

- [ ] Page title React App -> AURAI in public/index.html
- [ ] Remove dead footer links in Home.jsx
- [ ] Extract RiskBenefitWidget, ExplainabilityCard, OutcomeSimulationCard out of History
- [ ] Fix age field not updating in PUT /me (users.py line 92)
- [ ] Email format validation on Login + Register
- [ ] Replace print() debug with logging.debug()
- [ ] Named constants for magic numbers (0.06, 0.65, 0.45)
