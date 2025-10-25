# 🫧 Electrochemical Bubble Tracking — Bright & Dark Detection Framework

### 🎯 Overview

This repository presents a computational framework for detecting, tracking, and quantifying gas bubbles generated during electrochemical reactions.  
The system is designed to handle complex visual environments — including bright reflective bubbles and dark shade-like bubbles — using adaptive image processing and trajectory-aware tracking algorithms.

The tracking programs are developed and optimized to analyze electrochemical gas evolution on three electrode systems:
- Bare Ti metal  
- TiO₂ nanotubes (TNT)  
- Carbon-doped TiO₂ nanotubes (CTNT)  

Each dataset corresponds to 30-second electrochemical experiments in Na₂SO₄ electrolyte under constant applied potentials of 5 V, 10 V, 15 V, and 20 V.

---

## 🧠 Framework Description

| Mode | Optimized for | Core Algorithm | Output |
|------|----------------|----------------|---------|
| ✨ Bright Bubble Tracker | Reflective or shiny bubbles | Threshold + trajectory mapping | bright_bubbles_per_second.csv, annotated images |
| 🩶 Dark Bubble Tracker v5-ID | Diffused or shadow-type bubbles | Background subtraction + motion-aware matching + ID locking | dark_bubbles_new_vs_left_v5.csv, trajectory CSVs |

Both frameworks perform:
- Frame-wise bubble detection  
- Persistent ID-based tracking  
- Identification of newly generated and leaving (upward-moving) bubbles per second  
- Generation of combined plots and annotated visual sequences

---

## 🎥 Experimental Video Dataset (Flimora Series)

All raw experimental videos are hosted on Google Drive (due to size limits) and correspond to electrochemical bubble evolution under different potentials and electrode types.

🔗 Dataset Link — Flimora Bubble Tracking Series:  
https://drive.google.com/drive/folders/14Lx6HuAEkTNryvftkNRynNMiNJ658CQs?usp=drive_link

---

### 📁 Folder Structure

flimora/
 ├── 5v/
 │    ├── Bare_Ti_5v_na2so4.mp4
 │    ├── TNT_5v_na2so4.mp4
 │    └── CTNT_5v_na2so4.mp4
 ├── 10v/
 │    ├── Bare_Ti_10v_na2so4.mp4
 │    ├── TNT_10v_na2so4.mp4
 │    └── CTNT_10v_na2so4.mp4
 ├── 15v/
 │    ├── Bare_Ti_15v_na2so4.mp4
 │    ├── TNT_15v_na2so4.mp4
 │    └── CTNT_15v_na2so4.mp4
 └── 20v/
      ├── Bare_Ti_20v_na2so4.mp4
      ├── TNT_20v_na2so4.mp4
      └── CTNT_20v_na2so4.mp4

Each folder represents a specific voltage condition, and the contained videos correspond to the three electrode types used in the electrochemical bubble dynamics study.

---

## ⚙️ Usage Instructions

1. Clone this repository or open it in Google Colab.  
2. Download required videos from the Drive link.  
3. Update the file paths in the code:
   BASE_DIR = "/content/drive/MyDrive/ColabNotebooks/Bubbles_Program/flimora/10v"
   VIDEO_NAME = "CTNT_10v_na2so4.mp4"
4. Run:
   - bright_bubble_tracker.ipynb → for shiny bubbles  
   - dark_bubble_tracker_v5.ipynb → for diffused/dark bubbles  
5. Outputs include:
   - Annotated image frames  
   - Trajectory and per-second CSVs  
   - Combined generation vs. leaving bubble plots

---

## 📊 Output Files Generated

| Output | Description |
|--------|-------------|
| bubbles_new_vs_left_combined.png | Combined plot of new and leaving bubbles per second |
| bubbles_per_second.csv | Time-series of bubble generation |
| dark_bubble_tracks_v5.csv | Detailed bubble trajectories |
| /annot_dark_v5/ or /annot_bright_v3/ | Annotated frame sequences |

---

## 🧩 Key Features

- Automatic frame extraction from videos  
- Adaptive background modeling (for dark bubbles)  
- ID-based bubble tracking with motion awareness  
- Detection of bubbles leaving electrode surface  
- Handles both shiny (bright) and diffused (dark) lighting conditions  
- Generates publication-ready plots and CSVs  

---

## 📚 Citation and References

If you use this repository or dataset in your work, please cite the following relevant study:

**Zhao, X.; Ren, H.; Luo, L.**  
*Gas Bubbles in Electrochemical Gas Evolution Reactions.*  
**Langmuir** 2019, *35*(16), 5392–5408.  
https://doi.org/10.1021/acs.langmuir.9b00138  

---

## ✨ Author & Development Note

This repository was developed as part of a computational framework for electrochemical bubble dynamics analysis, implemented by Aparna Markose at Pondicherry University, India.  
The code was designed, optimized, and validated in Google Colab, integrating automated image processing and motion-aware trajectory detection.
