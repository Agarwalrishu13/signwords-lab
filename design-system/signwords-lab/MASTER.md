# SignWords Lab interface

## Brief

Design for the user's laptop presentation, especially 1366 × 768 and 1440 × 900.
Keep collecting, recording, training, storage, detection and speech behavior.
The user rejected the sage theme and repeated rounded containers. Do not restore
those patterns or add a marketing landing page in front of the tool.

## Visual direction

A camera-led studio with a slim navigation rail, open vocabulary list and controls
placed directly below the camera. Sections are grouped with space and occasional
dividers. Only actual controls and the media viewport have rounded boundaries.

- Canvas: #fafbfe
- Navigation rail: #f0f2f7
- Text: #202737
- Secondary text: #606b80
- Border: #dce1eb
- Cobalt action: #3456d6
- Typeface: self-hosted Manrope, medium and bold for hierarchy
- Radius: 8px controls, 14px camera viewport, no section containers
- Motion: brief interaction feedback; respect reduced motion
- Theme: follow device preference, using semantic CSS variables

The camera-off photograph is decorative and is removed when the camera becomes
ready. Camera video and its landmark overlay use matching containment so the full
camera frame remains visible. Do not add a fake webcam preview or sample metrics.

## Structure

    navigation rail | screen title and description
                    | camera surface          vocabulary / model / speech
                    | camera controls         open controls and content
                    | recording controls      dataset tools / results

Training architecture and log are native disclosures. Settings, learning curves
and evaluation stay accessible within the Train model screen. Keep all current
element IDs, navigation labels, form order and event wiring.

## References

UI UX Pro Max provided the Minimalism & Swiss Style recommendation and interaction,
contrast, responsive desktop and semantic-token checks. Its marketing page pattern
and suggested multi-accent palette were not appropriate for this product brief.
The frontend-design skill informed the subject-specific photograph and the removal
of repeated containers. User instructions take precedence over generic skill defaults.

Source: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill
