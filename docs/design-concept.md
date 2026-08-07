# BdSL Vision Design Concept

## Chosen Direction

BdSL Vision uses a premium scientific AI interface: dark neutral background, quiet panels, high contrast type, and one cyan-blue signature accent for camera and recognition actions. The application opens directly on the recognition workspace instead of a marketing page.

## Screens

1. Recognition desktop screen
   - Sticky top navigation with BdSL Vision wordmark.
   - Large 16:9 camera stage with CSS-only mirrored preview.
   - Right-side recognition control stack with Start, Stop & Predict, Clear.
   - Recording rail overlays the camera with timer and max-duration progress.
   - Privacy note remains visible but secondary.

2. Prediction-result state
   - Predicted Bangla sentence is the dominant visual element.
   - Result actions: Copy Sentence, Try Again, download JSON.
   - Diagnostics stay below the result and use restrained metric tiles.
   - Confidence is explicitly labeled as uncalibrated.

3. Mobile recognition screen
   - Camera remains first.
   - Controls stack below with full-width touch targets.
   - No horizontal overflow; Bangla text wraps with Noto Sans Bengali.

4. Model page
   - Runtime details come from `/api/model/info`.
   - Pipeline visualization:
     Webcam Video -> MediaPipe -> 1662-D Landmark Sequence -> Raw + Velocity Streams -> Region Projection -> Learned Fusion Gate -> Depthwise Temporal Conv -> Transformer Encoder -> Word-Level CTC -> Bangla Sentence.

## Design System

- Background: near-black neutral, light mode available.
- Surfaces: flat research-grade panels with 1px borders.
- Accent: cyan-blue for active recognition and model status.
- Typography: Inter for UI, Noto Sans Bengali for Bangla output.
- Radii: 8px maximum for cards, buttons, panels, and controls.
- Motion: short reveal transitions only; reduced-motion respected.
- States: ready, recording, warning, loading, success, error.
- Camera treatment: unframed full camera area inside a single working stage; no decorative nested cards.

