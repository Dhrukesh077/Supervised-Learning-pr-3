# 🎥 Video Explanation Guide (5–10 minutes, face + screen)

**Rubric (Component B – 3 marks):** webcam picture-in-picture **and** the Jupyter Notebook on screen, explaining concepts **verbally while running** the notebook (do not just execute cells silently).
Record with **OBS Studio** (Scene → add *Display Capture* + *Video Capture Device* (webcam, small corner)). Upload to **Google Drive (share: "Anyone with the link")** or **YouTube – Unlisted**, then paste the link in `README.md`.

> Tip: do *Kernel ▸ Restart & Run All* once before recording so you know it runs clean, then record while stepping through the cells. Speak in your own words – the points below are prompts, not a script to read.

| Time | What to show | What to say |
|---|---|---|
| 0:00 – 0:30 | Webcam + title cell | Your name, project name, goal: early-warning classifier for high-risk bank customers. |
| 0:30 – 1:30 | **Part A** theory | Logistic Regression & sigmoid; Type-I vs Type-II error (false alarm vs missed risky customer); why accuracy is misleading when only ~12 % are high-risk. |
| 1:30 – 3:00 | **Part B** | Features vs target; missing values (~2–5 % in 7 columns); **stratified split** keeps 12 % share; **KNN Imputer** (scale first, k=5, fitted on train only → no leakage). |
| 3:00 – 4:00 | **Part C** | Baseline Logistic Regression; read the confusion matrix; define precision, recall, F1; point out TN/FP/FN/TP and which are Type-I/Type-II. |
| 4:00 – 5:30 | **Part D** | Dummy model: 88 % accuracy but 0 recall (accuracy paradox); under-/over-sampling, SMOTE (interpolates), ADASYN (focuses on hard cases); before/after table; the reduced-signal experiment shows recall ↑, precision ↓. |
| 5:30 – 6:30 | **Part E** | Decision tree overfits (train 100 % vs test lower, depth curve); Random Forest generalises better (bagging). |
| 6:30 – 7:30 | **Part F** | RandomizedSearchCV → GridSearchCV; why F2 scoring (recall counts double); tuned vs untuned. |
| 7:30 – 8:30 | **Part G** | ROC curves, AUC meaning (ranking quality across thresholds); final-model rule: **minimise false negatives**; mention the cross-validation caveat (hold-out is perfect, CV recall ≈ 0.975). |
| 8:30 – 9:30 | **Streamlit app** (browser) | Predict one customer, move the threshold slider, upload a CSV, show the performance tab. |
| 9:30 – 10:00 | Wrap-up | Key learnings, limitations (data very separable → validate on real data), next steps. |
