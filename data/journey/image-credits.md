# Image credits and provenance

Checked September 20, 2026. Publication should retain the credits and license links below. Sources do not imply endorsement of 1840.

## Original teaching visuals

**assets/plaque-rupture-to-muscle-illustration.png** — Original AI-assisted teaching illustration created for 1840 with the built-in `image_gen.imagegen` tool. Conceptual anatomy; not clinical imaging, not to scale. Final prompt saved in `production/anatomy-image-prompt.txt`. Reviewed for wall-versus-lumen placement, focal cap disruption, clot continuity, and downstream myocardial territory. It is not a patient-specific reconstruction.

**assets/ecg-normal-versus-st-elevation.png** and **.svg** — Original deterministic ECG teaching plot. Reproducible source: `production/render-ecg.mjs`. Idealized waveforms; not sampled patient data and not a diagnostic calibration tool. The ST-segment comparison and clinical context are supported by [NICE CG95, section 1.2.2](https://www.nice.org.uk/guidance/CG95/chapter/recommendations#resting-12-lead-ecg).

## Published human tissue

**assets/plaque-rupture-human-carotid-figure-2.jpg** — Unchanged embedded Figure 2 extracted from page 8 of:

Truong M, Dreier T, Wassélius J, Sundius L, Persson A, Lovric G, Bonnin A, Goncalves I, Bech M. *Sub-micrometer morphology of human atherosclerotic plaque revealed by synchrotron radiation-based μCT—A comparison with histology.* PLOS ONE 17(4): e0265598 (2022).

- [Original article and licensing statement](https://doi.org/10.1371/journal.pone.0265598)
- [Figure 2](https://doi.org/10.1371/journal.pone.0265598.g002)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- [Accessible copy of the published PDF](https://pdfs.semanticscholar.org/db47/e2993091431e419e8ce59ac9a3e20f6aa4eb.pdf)
- Specimen: human carotid plaque obtained at endarterectomy. Panels A–C: synchrotron micro-CT; D–F: histology with glycophorin A staining. Original marks and scale bars retained.
- Processing: embedded JPEG extracted with pypdf; no generative processing, color adjustment, cropping, relabeling, or changes to pixels.

Suggested short credit: “Truong et al., PLOS ONE (2022), Figure 2, CC BY 4.0. Human carotid tissue; reproduced unchanged.”

## Real ECG recordings

**assets/ecg-reference-recording.jpg** — [12 lead ECG of a 26 year old male](https://commons.wikimedia.org/wiki/File:12_lead_ECG_of_a_26_year_old_male.jpg). Creator: MoodyGroove. Released by creator into the public domain. Training-session recording, with identifying fields already obscured in the source. Its automated normal interpretation is marked unconfirmed. Reproduced unchanged; avoid claiming independent clinical adjudication.

**assets/ecg-inferior-rv-infarction.jpg** — [ST Segment Elevation Myocardial Infarction Unlabeled](https://commons.wikimedia.org/wiki/File:ST_Segment_Elevation_Myocardial_Infarction_Unlabeled.jpg). Creator of original recording image: James Heilman, MD; derivative by Andrewmeyerson. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Original source record](https://commons.wikimedia.org/wiki/File:Inferior_and_RtV_MI_with_PVC_15_lead.jpg). Source describes inferior and right ventricular infarction; 15-lead recording with a premature ventricular beat. Reproduced without further alteration. Keep this asset under its existing license; any future adapted version must use a compatible share-alike license.

These ECGs are independent examples from different people, not a serial clinical case and not linked to the histology specimen. Their lead layouts differ.

## Linked, not reproduced

[UAB PEIR image 00007056, Peter Anderson](https://peir.path.uab.edu/library/picture.php?/6950): H&E section described as a ruptured right coronary atherosclerotic plaque with thrombus. The viewed page did not establish a reproduction license, so the draft links to it as further study rather than incorporating it as a publication asset.
