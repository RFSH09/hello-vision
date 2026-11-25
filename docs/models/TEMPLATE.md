# Model card: <name> <version>

- **Role:** detector | embedder | other
- **Base model and license:** (code and weights; record in docs/LICENSES.md)
- **Training data:** dataset cards and versions
- **Training config / seed / commit:**
- **Metrics:** held-out sessions; per-SKU top-1, detection precision/recall, exact-cart accuracy
- **Export:** ONNX opset, input size, quantisation
- **Known limits:** lighting, SKUs, occlusion cases that fail
- **Approved for:** simulation | shadow | production (needs go/no-go sign-off)
