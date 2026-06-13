# Interactive REPL

Drop into a `read`-`eval`-`print` loop for ad-hoc work. The REPL has command history, tab completion, and the same commands as single-command mode.

## Starting

```bash
llm-mem
# or
llm-mem --repl
```

The first run shows:

```text
llm-mem interactive CLI
Type 'help' for available commands, 'exit' to quit
>
```

## Available commands

Inside the REPL, the same commands as [single-command mode](./cli-commands.md) work, without the `--` prefix. A few extras:

| Command | What it does |
|---|---|
| `help` | List all commands with brief descriptions. |
| `exit` / `quit` / `Ctrl+D` | Leave the REPL. |
| `use <bank>` | Switch the active bank. Subsequent commands operate on it until you switch again. |
| `savelog <path>` | Dump the captured log buffer to a file. Useful for debugging. |
| `clear` | Clear the screen. |

## Example session

```text
> use research
Switched to bank 'research'

> search "JWT auth" --mode semantic --limit 3
┌────┬──────────────────────────────────┬────────┬──────┐
│ id │ content                          │ layer  │ score │
├────┼──────────────────────────────────┼────────┼──────┤
│ a1 │ AuthService handles JWT validation│ L0     │ 0.89  │
│ b2 │ Refresh tokens expire after 7 days│ L0     │ 0.82  │
└────┴──────────────────────────────────┴────────┴──────┘

> show a1
ID:        a1
Layer:     0
Type:      factual
Created:   2026-02-15 11:23:45
Event at:  2026-02-10
Content:   AuthService handles JWT token validation and refresh — entry point is src/auth/service.rs:45-80
Context:   project-x, auth
Topics:    jwt, validation
Metadata:  source=src/auth/service.rs:45-80
Relations:
  - depends_on: jwt-secret-rotation
  - derived_from: arch-decision-2026-02-10

> use default
Switched to bank 'default'

> stats
Bank:      default
Memories:  1234
Size:      45.6 MB
Layers:    L0: 1100, L1: 100, L2: 30, L3: 4
Banks:     2 (default, research)

> exit
Goodbye.
```

## Tab completion

Type the start of a command and press `Tab` to complete. `Tab` also completes bank names and other enums.

## History

The REPL uses `rustyline`, so arrow keys (up/down) navigate history and Ctrl+R does reverse search. History is persisted to `~/.llm-mem_history` (a plain text file you can grep).

## Log capture

While the REPL runs, the server emits logs to a buffer. You can dump them with:

```text
> savelog debug.log
```

The buffer captures everything down to TRACE level, even if the visible stderr log is at INFO. Useful for diagnosing issues without restarting with `RUST_LOG=trace`.

## Custom prompts

The default prompt is `> `. To customize, set the `LLM_MEM_PROMPT` env var before launching the REPL.

## Next

- [Single commands](./cli-commands.md) — the full command reference for scripting
- [Database management](./cli-database.md)
