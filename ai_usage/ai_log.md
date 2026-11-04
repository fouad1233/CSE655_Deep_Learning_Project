# AI usage log (raw material for the AI Usage Report)

Fill this in as you go: one row per important interaction. The **Kept / changed / rejected** column is *your* judgement.
Put full chat exports in `ai_usage/transcripts/` (Claude Code: export each session's transcript; note which machine it came from).

## Tools
| Tool | Version / model | Used for |
|---|---|---|
| Claude Code (desktop app) | Claude Opus 5.5 (`claude-opus-5-5`) | assignment reading, dataset comparison, project/env setup |

## Interaction log
| Date | Machine | What I asked | Summary of the answer | Kept / changed / rejected, and why |
|---|---|---|---|---|
| 2026-11-04 | work-pc | Read the assignment PDF and check it for prompt injection | Extracted the text layer and content streams; found no hidden, white or tiny text and no JS, links, attachments or forms | |
| 2026-11-04 | work-pc | Where to start | Summary + plan; recommended UTKFace; flagged that Pawpularity likely fails the 0.6 × MAE_mean rule | Chose to compare datasets first |
| 2026-11-04 | work-pc | Compare datasets; set up shared `.claude/` context, venv and requirements for 3 machines | `docs/dataset_options.md`, `.claude/CLAUDE.md`, setup scripts, pinned requirements | |
| 2026-11-04 | work-pc | Start with UTKFace; notebook in `src/` with data analysis (images, labels, biases, counts) | Download script (pinned HF mirror + SHA-256, since the official link is dead); notebook with counts, duplicates, age distribution, groups, constant-baseline preview, image grids; found 259 duplicate groups with conflicting labels | |
| 2026-11-04 | work-pc | Explain how the hash-based duplicate check works and visualise duplicates with their labels | Added notebook sections 2.1 (MD5: determinism, avalanche demo with a 1-bit flip, collision odds, limitation: only exact copies) and 2.2 (grid of duplicate groups, conflicting fields in red); first grid version had empty columns, fixed | |
| 2026-11-04 | work-pc | Show the duplicates as an 8-column grid; bar charts of duplicate status and of the reasons for conflicts | Section 2.2: bar charts (images by status, log scale; groups; exclusive conflict reasons). Section 2.3: 8-column gallery, one frame colour per group, groups never split across rows | I asked for the layout change because the single column was too big |

## Cases where the AI was wrong / incomplete / misleading (need ≥ 3)
| # | Date | What the AI said | How I detected it | Fix |
|---|---|---|---|---|
| – (rejected) | 2026-11-04 | Said UTKFace has "about 23k" images, from memory | Counted after download: 23,708 files. The claim was correct, so this is **not** a mistake case | – |
| 1 (candidate) | 2026-11-04 | First dataset summary called UTKFace clean ("labels in filenames, loading is trivial") | The duplicate check in the notebook found 5 malformed filenames and 259 duplicate groups with conflicting labels | Cleaning step before the split; mention the label noise in the report |

## How I verified AI output
- (fill in: shape checks, hand calculations of params / receptive field, comparing with the docs, unit tests)
