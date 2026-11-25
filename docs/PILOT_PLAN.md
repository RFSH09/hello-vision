# Pilot plan: ONE test fridge, shadow mode only

Status: plan only. Nothing here has started. **No real charging, no lock control, no changes to the existing system.**
The module runs beside the current system, receives the same door sessions, and its results are only compared, never billed.

## Preconditions (all must be true before installing anything)

- [ ] Hardware questions answered ([HARDWARE_QUESTIONS.md](HARDWARE_QUESTIONS.md)): camera access (RTSP/USB or our own
      cameras), how the door open/close signal can be read (read-only tap, not control), power and internet for the box.
- [ ] A trained detector and embedder exist, with a model card, and passed the offline check on team-recorded data
      (not only synthetic): per-SKU top-1 reported, exact-cart accuracy on scripted sessions reported.
- [ ] Privacy: lawyer review of [PRIVACY.md](PRIVACY.md) done; face blur validated on consented footage; retention set;
      decision on where evidence is stored (inside Uzbekistan); signage / consent wording for the pilot fridge agreed.
- [ ] Supplier: confirmed whether purchase videos and recognized carts can be exported, and in what format; agreement that
      reading the camera stream/door signal does not breach the supplier contract.
- [ ] The fridge owner (location) agrees to the pilot.
- [ ] The go/no-go gate numbers in [GO_NO_GO.md](GO_NO_GO.md) are confirmed by the owner.

## Phases

| Phase | What | Duration | Exit criterion |
| --- | --- | --- | --- |
| P0 Bench | Spare fridge, scripted sessions recorded by our own team (protocol in [DATA_COLLECTION.md](DATA_COLLECTION.md)); train and evaluate; fix the worst failures | until offline gate met | exact-cart accuracy and false-charge rate on held-out scripted sessions meet targets |
| P1 Replay | If the supplier exports history: run our module on past purchases (weak labels), review disagreements by hand | 1-2 weeks | disagreement causes understood; label corrections fed back |
| P2 Install | One fridge: edge box taps the camera stream and the door signal read-only; module runs, results stored locally, evidence blurred | 1 day | health, metrics and logs verified; nothing in the existing system changed |
| P3 Shadow | Run for at least 14 days; weekly report from `eval.shadow` and `eval.run`; reviewers check every disagreement and every `needs_review` | >= 14 days | gate met on shadow data |
| P4 Decision | Go/no-go meeting using the checklist | 1 day | written decision |

## Operations during the pilot

- **Kill switch:** `systemctl stop hello-vision` (or unplug the box). The existing fridge behaviour is unaffected, because
  we only read.
- **Data:** evidence clips are blurred before writing, kept 30 days, stored locally. Weekly export of anonymised results
  (session id, cart, flags) to the shared report; no video leaves the fridge unless a reviewer needs a specific clip.
- **Roles:** owner (decisions), engineer on call (service, models), reviewer (checks clips daily during the pilot).
- **Weekly report:** sessions, agreement rate, review rate, every disagreement with clip, latency p95, crashes/restarts,
  queue depth, privacy incidents.
- **Stop conditions:** any privacy incident; the box interfering with the fridge or the existing system; repeated crashes
  (more than 3 unplanned restarts per day); owner request.

## Risks to watch

- Real products and lighting differ from everything tested so far; expect the first shadow week to show many failures.
- Look-alike SKUs (bottles, cans, bars) are the main accuracy risk; the embedding model must be fine-tuned and validated on them.
- The current system's carts are not ground truth; disagreements need human labels before they count against us.
- 14 days at one fridge is about 150-200 sessions if usage is typical, so a 0.2% false-charge limit cannot be demonstrated
  statistically from one fridge. Plan to extend to more fridges or longer before any charging decision.
