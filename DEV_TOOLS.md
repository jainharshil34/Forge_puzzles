# DataForge - Developer Tools & Diagnostic Workbench

This document outlines the internal diagnostic tools included in the repository.

---

## Internal Diagnostic Workbench (`web/`)

The [`web/`](file:///c:/Users/Harshil%20Jain/Desktop/dataforge/web) directory contains a lightweight, plain HTML/CSS/JavaScript workbench used for rapid backend validation, model sanity-checking, and raw inference debugging during model training and evaluation.

> [!IMPORTANT]
> **`web/` is an internal developer diagnostic tool only. It is NOT the submission artifact.**
>
> The official submission surface is the **Next.js 16 application located in [`src/`](file:///c:/Users/Harshil%20Jain/Desktop/dataforge/src)**, which features the full design system, IBM Plex Mono / Inter typography, fluid synaptic memory spring physics, diff overlays, interactive controls, and live telemetry.

---

### Purpose & Diagnostic Value of `web/`

During model development, the lightweight workbench in `web/` allows researchers to:
1. **Sanity-check PyTorch checkpoints**: Quickly test raw outputs from `context_model/checkpoint.pt` and `optimization_model/checkpoint.pt` without frontend build dependencies.
2. **Inspect raw tensors & activation vectors**: Verify step-by-step recurrent state traces ($h_0 \dots h_N$) and inner gradient loss trajectories.
3. **Verify REST API endpoints**: Directly test `/api/puzzles`, `/api/predict`, `/api/rules`, and `/api/compare` served by `serve.py`.

### How to Run the Diagnostic Workbench

The diagnostic workbench is served directly by the Python backend server:

```bash
# Start backend server with diagnostic workbench
python serve.py 8000
```

Access the internal diagnostic workbench at: `http://localhost:8000`

---

## Official Submission Artifact (`src/`)

The public-facing research instrument and official submission artifact is built with **Next.js 16 (App Router, Turbopack, TypeScript)**:

```bash
# Install frontend dependencies
npm install

# Run the official Next.js submission application
npm run dev
```

Access the official application at: `http://localhost:3000` (or the deployed public artifact URL).

---

## Deployment Configuration

- Next.js builds strictly from `src/` and does not include `web/` in the production bundle.
- All public artifact URLs and production deployments point exclusively to the Next.js application in `src/`.
