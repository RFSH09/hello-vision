# Hardware questions

Status: partially answered. The core stays hardware-agnostic. We plan for both
(a) reusing the existing cameras and (b) our own cameras if the existing ones are closed.

| Topic | Known | Still open |
| --- | --- | --- |
| Cameras | 2 x 1080p shelf cameras + 1 separate 24/7 security camera | models, FPS, connection (USB / RTSP / MIPI), exact angles and mounting - to be inspected on a spare fridge |
| Lock | electronic, fire-resistant, rated 500,000 cycles, driven by the fridge main computer | control interface (relay / GPIO / serial / vendor board) |
| Door sensor | - | type and how it is read |
| Main computer | an integrated computer currently runs the recognition; the server is outside Uzbekistan | whether it can be reused or must be bypassed |
| Internet | every fridge is online (monitoring, remote door opening via the web app) | bandwidth / stability |
| Power | standard mains; fridge has climate control | spare capacity for an edge box, thermal limits |
| Data export | - | can purchase videos and recognized carts be exported, and in what format (asked of the supplier) |
| Edge budget | draft: <= 2.5 mln so'm (~$200) per fridge | to be confirmed by the owner |
