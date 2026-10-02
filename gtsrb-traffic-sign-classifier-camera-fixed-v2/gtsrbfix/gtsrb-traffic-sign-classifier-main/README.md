# Traffic Sign Recognition AI — Camera Accuracy Fix

This version preserves the existing `traffic_sign_model.keras` GTSRB classifier and improves the camera pipeline.

## What changed
- Automatic camera sign localization using red/blue/yellow color masks + contours.
- Padded square candidate crops before CNN classification.
- Multiple candidate regions are scored; the best candidate is selected.
- Top-1 confidence AND Top-1/Top-2 margin are required before a result is treated as confirmed.
- Low-confidence/ambiguous results do not trigger voice alerts or history entries.
- Camera UI shows the exact crop sent to the CNN and the detected bounding box.
- Centered-sign mode remains available as a fallback.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Camera testing
Use one clear traffic sign at a time. Keep it large enough to occupy a useful part of the frame, avoid glare, and make sure the complete sign is visible. The localizer is designed to help with common red/blue/yellow GTSRB signs; it is not a full road-scene object detector.

## Important limitation
The existing model is a **classifier**, not a detector. This update improves localization without retraining the model, but it cannot guarantee perfect recognition in every road scene. A dedicated YOLO-style detector trained on traffic-sign bounding boxes is the next step for robust small-sign detection anywhere in a scene.
