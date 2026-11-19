"""Export a Claude Code JSONL session to readable Markdown (user + assistant text, tool calls as one line).

Usage: python scripts/export_chat.py SESSION.jsonl OUT.md [--list] [--skip i,j,k-l] [--title "work PC, 2026-11-04/05"]
  (Claude Code keeps session files in ~/.claude/projects/<project-folder>/<session-id>.jsonl)
  --list : print the user turns with their index (to choose what to skip)
  --skip : user-turn indices whose whole exchange (user turn + the replies until the next user turn) is left out
"""
import json, re, sys

path, out = sys.argv[1], sys.argv[2]
LIST = "--list" in sys.argv
TITLE = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else "Claude Code session"
skip = set()
if "--skip" in sys.argv:
    for part in sys.argv[sys.argv.index("--skip") + 1].split(","):
        a, _, b = part.partition("-")
        skip.update(range(int(a), int(b or a) + 1))

SR = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def user_text(content):
    if isinstance(content, str):
        return SR.sub("", content).strip(), False
    texts, img = [], False
    for b in content:
        if b.get("type") == "text":
            texts.append(SR.sub("", b["text"]))
        elif b.get("type") == "image":
            img = True
        elif b.get("type") == "tool_result":
            return None, False
    return "\n".join(t for t in texts).strip(), img


turns = []          # list of dicts: role, text, ts
for line in open(path, encoding="utf-8"):
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    if e.get("isSidechain") or e.get("isMeta") or e.get("isCompactSummary") or e.get("isVisibleInTranscriptOnly"):
        continue
    t, msg = e.get("type"), e.get("message") or {}
    ts = (e.get("timestamp") or "")[:16].replace("T", " ")
    if t == "user":
        txt, img = user_text(msg.get("content", ""))
        if txt is None:
            continue
        if not txt and not img:
            continue
        if txt.startswith("<command-") or txt.startswith("<local-command") or "This session is being continued" in txt:
            continue
        if img:
            txt = (txt + "\n\n*[image attached]*").strip()
        turns.append(dict(role="user", text=txt, ts=ts))
    elif t == "assistant":
        for b in msg.get("content", []) or []:
            if b.get("type") == "text" and b["text"].strip():
                turns.append(dict(role="assistant", text=b["text"].strip(), ts=ts))
            elif b.get("type") == "tool_use":
                name = b.get("name", "")
                if name.startswith("mcp__ccd_turn__") or name.startswith("mcp__ccd_session__"):
                    continue
                inp = b.get("input", {}) or {}
                desc = inp.get("description") or inp.get("file_path") or inp.get("caption") or ""
                turns.append(dict(role="tool", text=f"{name}" + (f": {desc}" if desc else ""), ts=ts))

# number user turns and drop skipped exchanges
ui, keep, cur = -1, [], False
for tr in turns:
    if tr["role"] == "user":
        ui += 1
        cur = ui in skip
        if LIST:
            print(f"[{ui}] {tr['ts']}  {tr['text'][:150]!r}")
        tr["ui"] = ui
    if not cur:
        keep.append(tr)
if LIST:
    sys.exit()

with open(out, "w", encoding="utf-8") as f:
    f.write(f"# Claude Code session transcript ({TITLE})\n\n")
    f.write("Model: Claude Opus 5.5 (`claude-opus-5-5`) in the Claude Code desktop app. "
            "User and assistant messages are shown in full; tool calls are listed by name and description.\n\n")
    for tr in keep:
        if tr["role"] == "user":
            f.write(f"\n---\n\n## User ({tr['ts']})\n\n{tr['text']}\n\n")
        elif tr["role"] == "assistant":
            f.write(f"**Assistant:**\n\n{tr['text']}\n\n")
        else:
            f.write(f"> *tool: {tr['text']}*\n\n")
print("turns kept:", len(keep), "of", len(turns))
