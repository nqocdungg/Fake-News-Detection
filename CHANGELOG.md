# CHANGELOG

> Chi tiết đầy đủ xem tại `docs/CHANGELOG.md`

---

## [2026-06-21] Master plan & Phase 7.4 plan

- `docs/plan/master-plan.md` — cập nhật results thực tế Phase 4 + Phase 7.1–7.3 ✅
- `docs/plan/phase-07.4-model-training-welfake.md` — kế hoạch train 4 model trên WELFake
- Phase 7.1–7.3 đã chạy xong: 72,074 rows, splits 50451/10811/10812

## [2026-06-18] Phase 7 — WELFake Integration (7.0→7.3)

- Plan files: `phase-07-dataset2-welfake.md`, `phase-07.1/2/3-*.md`
- Notebooks: `10_welfake_preprocessing.ipynb`, `11_welfake_eda.ipynb`, `12_welfake_vectorization.ipynb`
- Chuẩn hóa nhãn WELFake về `REAL=0`, `FAKE=1` giống pipeline ISOT hiện tại
- Output preprocessing: `data/processed/preprocessed_welfake_full.csv`

## [2026-06-17] Phase 4 — SVM Training hoàn thành

- LinearSVC (C=1): Val Accuracy = **0.9869**, F1 = **0.9869**
- Model: `models/svm_model.pkl`
- Notebook: `notebooks/06_svm.ipynb`
- Báo cáo: `docs/SVM-walkthrough.md`

## [2026-06-17] Khởi tạo tài liệu

- `docs/plan/master-plan.md`, `AGENTS.md`
