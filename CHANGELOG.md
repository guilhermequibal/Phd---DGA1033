# Changelog

Records what changed, why, and the origin of the decision for every change
to a frozen or versioned component (see `CONFIG_POLICY.md`). Purely
code-cleanup commits with no effect on evaluation logic, output format, or
experiment parameters are not tracked here — only changes that could affect
comparability between runs.

## 2026-07-20 — Reorganized files into labelled folders

**What:** Grouped the project's data, instruments and outputs into three
labelled folders, leaving all executable code at the project root:
`01_frozen/` (envision_index.json, ISI Manual PDF, sample_project.pdf,
prompt_domain.txt), `02_adjustable/` (prompt_output_schema.txt, Document
template.xlsx), and `03_outputs/` (renamed from `results/`; holds every
`exp__*` run, manifests and logs). The transitional `config/` folder was
dissolved — its two files moved to `01_frozen/` and `02_adjustable/`
according to their frozen/adjustable status. `Experiment Cell 1.py` now
defines `FROZEN_DIR` / `ADJUSTABLE_DIR` / `OUTPUTS_DIR` (all data paths are
centralized there); the launcher, Cell 2.5 and `.gitignore` were updated for
the `results/` → `03_outputs/` rename. The generated `chroma_db/` index stays
at the root (gitignored, auto-rebuilt). A new `README.md` maps every file with
its edit-safety label.

**Why:** So the frozen / adjustable / output distinction from
`CONFIG_POLICY.md` is visible in the filesystem itself, making accidental
edits to frozen material far less likely. No file *content* changed and no
result was reclassified — this is a pure relocation. Prompt-file versions
(`prompt_domain` v1.0, `prompt_output_schema` v1.0) are unchanged, so runs
before and after this move remain comparable.

**Origin:** User request for a clearer folder structure that prevents
accidental changes, building on the frozen/adjustable model in
`CONFIG_POLICY.md`.

## 2026-07-16 — Split system prompt into domain logic + output schema

**What:** Replaced the single `rag_system_prompt.txt` with two versioned
files under `config/`: `prompt_domain.txt` (v1.0) and
`prompt_output_schema.txt` (v1.0). `Experiment Cell 4.py` now concatenates
them at load time and records both versions/hashes in
`experiment_manifest.json` and in every run's `_summary.json`. The original
file is kept at `archive/rag_system_prompt_v0_monolithic.txt` for reference
only and is no longer read by the experiment code.

**Why:** The single-file prompt conflated two different concerns — the
Envision evaluation logic (what is scored) and the JSON reporting format
(how it is reported). That made it impossible to iterate on output
formatting (to ease analysis) without risking an undocumented change to
scoring logic, and made prompt changes hard to audit or version
independently. See `CONFIG_POLICY.md` §3.

**Origin:** Internal methodology review formalizing the frozen/adjustable
component split (`CONFIG_POLICY.md`). No project results were reclassified as
a result of this change — the split is a bookkeeping/versioning change, not a
content
change: `prompt_domain.txt` v1.0 and `prompt_output_schema.txt` v1.0
together reproduce the prior monolithic prompt byte-for-byte (modulo the
version-header comments and the physical split point).

## 2026-06-04 (`8f0cf97`) — Added ISI Guidance Manual to RAG retrieval

**What:** Added the ISI Envision Manual PDF as a second indexed source
alongside the workbook, and updated the prompt accordingly, in the same
commit as unrelated resource-control changes.

**Why:** A domain expert identified that correct application of the
Envision standard requires the manual's applicability language, which the
workbook alone does not carry — independent of any test project's output.

**Origin:** Domain-expert guidance.

**Note:** Because this commit bundled a domain correction with unrelated
changes, no result is claimed *from* this transition — its role is to
*establish* the corrected baseline, not to *measure* an effect. Results
predating this commit are pilot runs against a mis-specified instrument and
are not part of the frozen scientific baseline.
