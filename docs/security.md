# Security

AgentBox is designed for **local research sandboxes**, not multi-tenant hostile
isolation. Docker is the trust boundary.

## Guarantees

| Control | Default behavior |
| --- | --- |
| Execution location | Tools and verifier run **inside** the container only |
| Host project mount | Not mounted by default; files injected via API |
| Path access | Jaileed under `/workspace` |
| Network | Disabled (`network_mode=none`) |
| Resource caps | CPU, memory, pids limits |
| Privileged mode | Never enabled by AgentBox |
| Docker socket | Must **never** be mounted into sandboxes |

## Residual risk

- Root inside the container still depends on kernel/Docker isolation quality.  
- Enabling network allows outbound traffic (exfil / supply-chain installs).  
- Generated tasks may contain unexpected setup commands — QC them.  
- Model providers see prompts and tool schemas (host-side HTTP).  

## Recommendations

1. Keep `network_disabled=True` unless setup truly needs package installs.  
2. Prefer baked images with preinstalled deps over open network + pip.  
3. Do not put secrets in `starter_files`, `setup_commands`, or container env.  
4. Run `agentbox prune` periodically to avoid disk fill from orphaned containers.  
5. Treat trajectory logs as potentially sensitive (code + tool outputs).  

## Threat model (local research)

**In scope:** accidental host path access via agent tools; runaway CPU/memory;
leftover containers.

**Out of scope:** defending against a malicious Docker daemon; multi-tenant cloud
isolation; browser or GUI escape.

## Related

- [Sandbox](sandbox.md) — path jail and resources  
- [Configuration](configuration.md) — limit knobs  
