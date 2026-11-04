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

## Cases where the AI was wrong / incomplete / misleading (need ≥ 3)
| # | Date | What the AI said | How I detected it | Fix |
|---|---|---|---|---|
| 1 (candidate) | 2026-11-04 | Said UTKFace has "about 23k" images, from memory and without a source; the official page only says "20k+" | The AI checked the official page in the next turn | Count the images after download and use that number (confirm or reject this case then) |

## How I verified AI output
- (fill in: shape checks, hand calculations of params / receptive field, comparing with the docs, unit tests)
