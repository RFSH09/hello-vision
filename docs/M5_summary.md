# M5 summary

FastAPI service (`/session`, `/health`, `/metrics`), signed idempotent backend client with a durable offline queue and backoff, evidence clips with face blur, retention, review UI with corrections, and the active-learning "to label" queue. Contract in [API.md](API.md), privacy in [PRIVACY.md](PRIVACY.md).

- 20 service tests, including a full session flow (video -> cart -> signed POST), offline -> online delivery, duplicate suppression, path-traversal rejection, review auth and HTML escaping.
- **Found while building:** OpenCV 5 no longer ships the Haar `CascadeClassifier`, so face blurring uses YuNet (MIT) loaded from a separate model file. The service refuses to start without it.
- **Not validated:** face detection on real faces (only that the model loads and returns nothing on a blank image), the review UI in a real browser (served HTML is tested, JavaScript is not), real Django backend (tested against a mock transport).
