# Go / no-go checklist before any customer is charged by our module

The numbers are PROPOSED and need the owner's confirmation. All of these must be true; one "no" means no-go.

## Accuracy (measured by `eval.run` / `eval.shadow` on SHADOW data, not synthetic)

- [ ] Exact cart accuracy on auto-charged sessions >= 99% sustained over 14 days
- [ ] False-charge rate < 0.2% (a customer billed for something not taken)
- [ ] Review rate < 3% of sessions
- [ ] Result < 5 s after door close at p95, **measured on the target edge device with the real models**
- [ ] Enough sessions that the numbers mean something: at 0.2% a few hundred sessions cannot prove the limit
      (about 1,500 error-free sessions are needed to show < 0.2% at roughly 95% confidence; more fridges or more days)
- [ ] Every disagreement with the current system reviewed by a human with the clip, and categorised
- [ ] Per-SKU confusion checked for look-alike groups (bottles, cans, bars); no group below the target
- [ ] Held-out scripted scenarios all pass: take 1, take 2, take and return, take nothing, fast hand, two hands, occluded, low light

## Safety and fallback

- [ ] `needs_review` sessions are NOT charged; the backend honours it (contract test in the Django project)
- [ ] Defined handling for review sessions: who reviews, how fast, what the customer sees
- [ ] Kill switch and rollback tested (stop the service, roll back the model with `ModelStore.rollback`)
- [ ] The existing system can stay in charge as fallback during the first charged weeks

## Privacy and legal

- [ ] Lawyer sign-off on [PRIVACY.md](PRIVACY.md) and data location
- [ ] Face blur validated; retention and access control to evidence in place
- [ ] Customer notice / terms updated if needed

## Operations

- [ ] HMAC secret provisioned per environment; backend verification and idempotency tested end to end
- [ ] Offline queue tested with a real internet outage
- [ ] Monitoring and alerts on `/metrics` (outbox depth, review rate, restarts)
- [ ] Model update and rollback rehearsed on the pilot fridge
- [ ] On-call owner named
- [ ] Docker image built and systemd unit run on the real device
- [ ] Dependency license scan done and recorded in [LICENSES.md](LICENSES.md)

## Decision record

| Date | Decision | Evidence (report links) | Signed by |
| --- | --- | --- | --- |
| | | | |
