# Data collection protocol (proposal; counts to be confirmed after the first baseline)

Recorded by our own team, with written consent, on spare (unused) fridges. Data stays in private storage under DVC
([DVC.md](DVC.md)); faces are never kept unblurred outside the recording PC's temporary folder.

## Rules

- Only team members who signed the consent form appear in the footage. No customers, no real purchases.
- One scripted session = door opens, the scripted action is performed, door closes. The recorder writes the session truth
  file (`<name>.truth.json`: cart and TAKE/RETURN actions with the approximate frame) immediately after each take.
- Vary the person, hand (left/right), speed, grip and which shelf/slot between sessions; vary the shelf layout between blocks.
- Never reuse a session for both training and evaluation: split by recording day (about 20% of days held out).

## Scenario matrix and target counts

| Scenario | Per SKU (30 SKUs) | Extra | Notes |
| --- | --- | --- | --- |
| Take 1, normal light | 12 | | |
| Take 1, low light | 4 | | lights dimmed / door-light off |
| Take 1, fast hand | 4 | | |
| Take 1, occluded (hand covers most of the product) | 4 | | |
| Take and return the same item | 5 | | |
| **Per-SKU subtotal** | **29** | | 870 sessions |
| Take 2 different SKUs | | 150 | include every look-alike pair at least 5 times |
| Take 2 of the same SKU | | 40 | |
| Take 3 items | | 30 | |
| Take nothing (open and close) | | 100 | about 4% of real openings do this; over-sample for robustness |
| Two hands / two people reaching | | 60 | |
| Swap: take A, return A, take B | | 50 | |
| Look-alike focus: take each look-alike in turn (bottles, cans, bars) | | 10 per group | Coca-Cola/Fanta/Fuse Tea style bottles; Red Bull sizes; Gorilla/18+/Adrenaline cans; Snickers/Twix/Bounty/KitKat |
| **Total** | | | about 1,250 sessions, roughly 30 hours of recording (about 4-5 person-days) |

For the detector, extract about every 5th frame and label boxes (`product`, `hand`) with CVAT; pre-annotate with the
current model to speed this up ([datatools.labeling](../hello_vision/datatools/labeling.py)). For the embedding model, crop
products by SKU (`training.build_embedding_dataset`).

## Checklist per recording day

- [ ] Consent forms signed, recording PC and storage location noted
- [ ] Camera positions and lighting recorded (photo of the setup without people)
- [ ] Sessions named `<date>-<block>-<nnn>`; truth files written and spot-checked
- [ ] Footage copied to private storage, local copies deleted, DVC updated, dataset card updated
