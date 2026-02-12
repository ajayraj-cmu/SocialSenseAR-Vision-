# Test Commands Reference — All Mask & Effect Types

Use these in the Unity Editor **Command** field (or speak them with "Hey Vibe" … "thank you") to exercise every effect and pattern.

---

## 1. Per-object effects (mask one thing)

| Effect   | Example commands |
|----------|------------------|
| **Blur** | `blur person` · `blur laptop` · `blur monitor` |
| **Dim**  | `dim person` · `dim laptop` · `dim monitor` |
| **Pixelate** | `pixelate person` · `pixelate laptop` |
| **Highlight** | `highlight person` · `highlight laptop` |
| **Outline** | `outline person` · `outline laptop` |
| **Color** | `make person blue` · `color laptop red` · `paint person green` |

**More color examples (object + color):**  
`make person yellow` · `color laptop orange` · `paint person purple` · `make person pink` · `color laptop cyan` · `make person magenta` · `color laptop gold` · `make person teal` · `paint person coral`

---

## 2. Inverted effects (effect on everything *except* the target)

| Command | What it does |
|--------|------------------|
| `blur background` | Blur everything except person (privacy blur). |
| `blur everything but person` | Same as above. |
| `blur everything but laptop` | Blur except laptop. |
| `dim everything except person` | Dim except person. |
| `pixelate everything but me` | Pixelate except person ("me" → person). |
| `highlight everything but laptop` | Highlight everywhere except laptop. |
| `make everything but person blue` | Color overlay everywhere except person. |

---

## 3. Full-screen effects (no mask, whole view)

| Command | What it does |
|--------|------------------|
| `dim everything` | Full-screen dim. |
| `dim the lights` | Often interpreted as dim (object or full-screen). |
| `make screen blue` | Full-screen color tint (blue). |
| `make screen red` | Full-screen red tint. |

---

## 4. Clear / remove

| Command | What it does |
|--------|------------------|
| `clear` | Remove all effects and prompts. |
| `reset` | Same as clear. |
| `stop blurring person` | Remove blur from person only. |
| `remove blur from laptop` | Remove blur from laptop. |
| `unblur person` | Remove blur from person. |

---

## 5. Utility

| Command | What it does |
|--------|------------------|
| `list` or `status` | Log active prompts and effects (server console). |
| `show` | Same as list. |

---

## 6. Suggested test sequence (copy-paste into Command field)

Run **Clear** first, then try in order:

```
clear
blur person
clear
dim laptop
clear
pixelate person
clear
highlight person
clear
outline person
clear
make person blue
clear
make person red
clear
blur background
clear
blur everything but laptop
clear
dim everything except person
clear
dim everything
clear
make screen blue
clear
```

---

## 7. Effect types summary

| Type       | Description |
|------------|-------------|
| **blur**   | Blur or heavy obfuscation (privacy). |
| **dim**    | Darken with semi-transparent black. |
| **pixelate** | Blocky censor-style pattern. |
| **highlight** | Bright/warm highlight overlay. |
| **outline** | Colored outline around mask. |
| **color**  | Solid or tint overlay (use color name). |

Supported **color names** (for `make X blue`, etc.):  
red, green, blue, yellow, cyan, magenta, white, black, gray, orange, purple, pink, brown, tan, lime, teal, navy, gold, silver, coral, violet, turquoise, emerald, crimson, light blue, dark blue, etc.

---

## 8. Targets that work well with SAM3

- **person** (always available if you're in frame)  
- **laptop** · **monitor** · **screen** · **computer** (aliased)  
- **phone** · **chair** · **couch** · **lamp** · **door** · **wall**  
- **background** → treated as “everything except person” (inverted blur)

If a target isn’t in the scene, the server may still add it as a prompt; the mask will appear when that object is detected.
