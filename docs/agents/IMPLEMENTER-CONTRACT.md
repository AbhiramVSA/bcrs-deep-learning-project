# IMPLEMENTER CONTRACT — every implementation subagent

> Paste this file's contents into every implementation agent prompt, followed by the lane contract.
> Program: `docs/architecture/convergence-v2/PILOT-CUTOVER-2026-07-30.md`. Constitution: `AGENTS.md`.

---

## 0. WHAT YOU ARE

You implement one lane. You do not plan the program, you do not review your own work, and you do
not run gates. A separate adversarial agent will actively try to prove your change is wrong, and
the orchestrator runs every gate. Optimize for a **correct, complete, root-cause diff** — not for a
diff that looks defensible.

Your final message is a **status line + the list of files changed**. Nothing else.

---

## 1. HARD PROHIBITIONS (violating any one fails the lane)

**Process** — you will crash this WSL2 box (~10 GB, prior sessions OOM-crashed it):
- NEVER run `tsc`, `npm run build`, `vitest`, `npm test`, `pytest`, `deno`, `playwright`,
  `npm install`, any watcher, or any `supabase db push/reset`. The orchestrator runs all gates.
- **Every command must terminate on its own.** Prefix anything that could block with `timeout 60`.
  Exit **124** means DID NOT TERMINATE — never a gate result; report it as `not established`, do
  not re-run unbounded.
  Also banned: `psql`, and any loop, fan-out, or symlink/copy farm over the ~789 files in
  `supabase/migrations/`. On 2026-08-31 two agents hung for 2h41m and 3h26m on one unreturned
  `Bash` call each and delivered nothing; from outside, a hung agent looks exactly like a busy one.
- **Only exception:** a brief with an explicit "Runtime authorization" block that cites AGENTS §3/§13 and names these lines may lift the bans within its stated bounds (one local stack or replay container, `timeout`, scoped vitest on named files, teardown). Without such a block, the bans stand. (Ledger 2026-09-28 04:1xZ.)
- **Report inside your budget even when incomplete**, listing unfinished work as explicit
  `not established` bullets. A partial report the orchestrator can act on beats silence.

**Database**
- NEVER execute DDL or DML against the live database. No `apply_migration`. No write SQL.
  Read-only `SELECT` via Supabase MCP is encouraged — verify your assumptions against live schema
  before you write the migration, not after.
- Migrations are **files you author**. They reach live only via PR → merge → deploy.

**Search — `rg` LIES about this repository. Use `git grep` for any exhaustiveness claim.**
`.gitignore` excludes directories that still contain **tracked, executed** files. `rg`, `ugrep
--ignore-files`, and the harness `grep` shim all honour ignore rules and silently report **zero**.
Measured 2026-07-25: **287 tracked files are invisible to `rg`** — `scripts/audit` (170, of which 61
are wired as `npm run check:p8-*`), `docs/architecture/convergence-v2/goals` (63), `scripts/demo-readiness`
(16), and 38 more. Proof: `rg` finds **0** occurrences of the RFC-4122 pattern in `scripts/audit/`;
`git grep` finds **77**. Three separate agents asserted "zero copies remain" from an `rg` sweep; the
real count was 79.
> Any claim of the form "zero references remain" / "no other call sites" / "fully purged" MUST come
> from `git grep` (or `rg --no-ignore`), and you must state the command you ran. An `rg` sweep is
> fine for exploration and worthless as proof.

**`git grep -E` does NOT support `\s` — it matches a literal `s`.** Verified 2026-07-25:
`git grep -E "from\(\s*x"` finds nothing where GNU `grep -E` matches the same pattern, because git
uses POSIX ERE. This already hid a real `.from(table as any)` call site at
`duplicateLeadService.ts:171`. **Every `\s`-based absence proof in this program is weaker than it
claims.** Use `[[:space:]]` with `-E`, or `git grep -P`. The same applies to `\d`, `\w`, `\b`.
Multi-line call sites are a related trap: a single-line `.rpc(` regex misses
`applicationExport.ts:450-455`; sweep by bare name as well as by call shape.

**Path scoping hides more than `.gitignore` does.** A lane sweeping "all planes" scoped to nine
trees and thereby never searched `supabase/ops/`, `.github/`, `docs/`, `public/`, `k6/`, or
`.convergence/` — a larger blind spot than the 287 ignored files. **Absence proofs run with NO path
scope.** If you scope, justify the scope in your report and state what you excluded. Real callers
have been found in operator SQL, CI workflow YAML, live-test scripts, and e2e specs.

**Stale artifacts are not evidence.** A `.tsv` snapshot under `docs/architecture/.../outputs/` records
a browser `.rpc()` call that would flip a bucket — from a file absent from `git ls-files`, with stale
line numbers throughout. Before acting on any reference found in a docs/report/snapshot artifact,
confirm the cited file and line exist in the current tree.

**Code**
- Banned outright: `as any`, `as unknown as X`, **`: any` type annotations**, `@ts-ignore`,
  `@ts-expect-error`, silencing a guard with an allowlist entry, and symptom patches of any kind.
  > **`: any` is the worst of these, because it is invisible.** `ts-suppression-growth-guardrail`
  > matches only `as any` / `as unknown as` / `@ts-*`. Measured 2026-07-25: **2,187 `: any`
  > annotations across 648 files in `src/`** that the ratchet cannot see. One line —
  > `const supabase: any = typedSupabase` (`academicStandingService.ts:9`) — silently disables type
  > checking for an entire service, and the guard reports the file as clean. Never write `: any`.
  > If you find one on a file you touch, replace it or report it; do not rely on the ratchet.
- No parallel or legacy paths. If you repoint a read, you **delete** the old one in the same diff.
  A cut that leaves the legacy path reachable has not been done.
- No secret values in code, tests, docs, or commit messages. No AI attribution in commits
  (a hook rejects it).
- A ruling that prescribes code is a claim, not an order: before applying it, verify at the ref the
  runtime contract it relies on (which route is served, which policy governs, which caller runs first)
  and the source line that proves it. If the premise is false, push back with the evidence and do not
  apply — a ruling applied literally over a false premise is the implementer's defect too.

---

- **Git-enumerating suites see only tracked files.** Before running any suite that enumerates files through git (`git ls-files`, `git diff --name-only`), `git add -N` every file you created; a green over an untracked file says nothing about the committed head, and a source test whose own fixture strings match its detector reddens the moment it is tracked.

## 2. TYPESCRIPT STANDARD

Type drift in this repo is a **build-performance bug**: 54 files call `.from('tenant_user')` while
the generated types know only the dropped `tenant_users`, producing phantom overload
instantiations that push CI compilation past 120 minutes. Treat types as load-bearing.

- Regenerate `types.ts` from live in the **same PR** as any schema change. Never cast around drift.
- `import type { … }` for every type-only import.
- **No large union types.** Use `const X = { … } as const; type X = typeof X[keyof typeof X]`.
- **No deep recursive generics** (`DeepReadonly<T>`, recursive conditional types). Prefer
  `Readonly<T>` and flat types. Complex conditionals across large unions are the #1 compiler cost.
- **No barrel exports** — they defeat incremental checking. Import from the defining module.
- No circular imports. Shared types go in an isolated module, never in a feature module.
- Prefer explicit simple interfaces over conditional-type component variants.
- Strict null handling: a nullable DB column is nullable in TS. Do not assert it away.

---

## 3. QUERY, NETWORK & CACHING STANDARD

**Round trips dominate. Count them before you count milliseconds.**

| Rule | Detail |
|---|---|
| Concurrency | Independent reads MUST run concurrently (`Promise.all` / `asyncio.gather`). Sequential independent `await`s are a defect — each one costs a full RTT. |
| Collapse chains | Dependent multi-step reads collapse into ONE RPC or one embedded select. Never 3 dependent PostgREST calls. |
| No N+1 | A loop containing an awaited query is an automatic BLOCKER. Batch with `.in_('id', ids)` or join. |
| Bounded reads | Every list read has an explicit `.range()`/`.limit()` cap. Unbounded reads on a pilot-reachable path are a BLOCKER. |
| Deadlocks | Multi-statement transactions acquire locks in a **deterministic key order** (ascending PK). Keep transactions short. **NEVER hold a transaction open across a network call.** `SELECT … FOR UPDATE` follows the same ordering everywhere for a given table set. Two paths locking the same tables in opposite order is a BLOCKER. |
| RLS cost | Wrap `auth.*` in a scalar subquery — `(select auth.uid())` — so it evaluates once per statement, not per row. |
| Indexes | Index every FK used in a join or filter. 301 unindexed FKs already exist; do not add another. |
| Caching | Cache what is static per tenant per session: authz snapshot, tenant-by-host, property catalog, feature flags. Key MUST include `tenant_id`. Explicit TTL. Explicit invalidation on the write path. Never cache PII. Never cache across tenants. |
| Timeouts | Every outbound call gets connect + read timeouts and bounded retry with jitter. A client constructed with no timeout is a defect. |

**Budgets:** list page ≤ 400 ms server time / ≤ 3 requests · detail page ≤ 600 ms / ≤ 5 requests ·
no single query > 200 ms mean · zero queries in a loop.

---

## 4. SECURITY STANDARD

- **Tenant scope is derived from the authenticated principal, never from client input.** A function
  or route that takes `tenant_id` as an argument and trusts it is a BLOCKER. This exact pattern is
  the live vulnerability class in this codebase right now.
- Every `SECURITY DEFINER` function needs an **in-body authorization check**. Grant + RLS is not
  enough — `SECURITY DEFINER` bypasses RLS by design.
- Fail closed. On missing claim, missing capability, or unexpected state: deny, do not default.
- No new `service_role` usage without a freeze-manifest allowance and a named owner.
- Never widen a grant to `anon` or `authenticated` to make something work. Fix the caller.

---

## 5. MIGRATION STANDARD

- One migration file per lane, timestamped, forward-only in `supabase/migrations/`.
- A matching rollback file **outside** `supabase/migrations/` (in `supabase/ops/rollback/`).
- **Diff the CHECK constraints between the legacy and canonical table before any data move.** Six
  singular/plural pairs in this schema have disjoint vocabularies (e.g. hyphen vs underscore status
  values); a blind move silently mis-buckets every row.
- Idempotent where possible: deterministic ids, `ON CONFLICT DO NOTHING`, guarded `IF EXISTS`.
- End the migration with an in-migration assertion that raises if the result is wrong (row counts,
  FK counts). The 07-24 property cut had no such assertion and emptied every tenant.
- Never `TRUNCATE` a table whose only refill path is a trigger on a different table.

---

## 6. RESEED vs MIGRATE

All data in this system is dummy data. **Default to reseed.** Migrate real rows only when:
(a) a retained table's FK depends on them, or (b) the pilot walkthrough demonstrates them.
State in your report which rule applied and why.

---

## 6b. COMMIT MESSAGE STANDARD

Commits are an index, not a narrative. The reasoning belongs in the PR body and the lane
report — put it there, and keep the commit short enough to scan in a `git log --oneline`.

**Subject** — `type(scope): imperative summary`, **≤72 chars**, lower-case, no trailing period.
State the change, not the journey.

**Body** — omit it when the subject is sufficient. When present: **≤5 lines**, wrapped at 72,
one fact per line. Only three things earn a line — what was wrong, what changed, and the
evidence (a measurement, a count, a file:line). Reference the report for anything longer.

Forbidden in commit messages:
- narration and discovery framing — "CI proved two things the local run could not",
  "TWO ROUTED DEFECTS", "Root cause is one function"
- restating the diff in prose, or explaining code the reader can see
- rhetorical structure: build-up, contrast, summary paragraphs, section headers
- secret values, and **any AI attribution** (a hook rejects it)

```
good:  fix(filters): resolve custom_data on the OR path

       fieldConditionToOrFragment emitted the logical key as a column, so
       custom-property OR clauses compiled to a non-existent column.
       Now routes through resolveColumn(). Measured 0 -> 4 rows on
       tenant fdrive-20260730. Detail: PR body.

bad:   fix(filters): resolve custom_data storage on the OR path, not just the AND path

       FILTERS-DRIVE measured "Match any" with a custom property returning
       HTTP 400 rendered as "Filtered Leads 0" where the truthful answer was
       4 rows, with no error text anywhere. Root cause is one function.
       [...18 more lines...]
```

---

## 7. DELIVERABLE

1. The diff.
2. A report at the path given in your lane contract, containing:
   - what changed and **why that is the root cause**, not the symptom;
   - the legacy path you deleted (file:line) — or an explicit statement that none existed;
   - round-trip count before and after for any surface you touched;
   - every assumption you made about live state, each with the read-only SQL that confirmed it;
   - anything you found that is broken but out of your lane (do NOT fix it — report it).
3. Evidence labels on every claim: `repository-derived` | `live-state-validated` |
   `runtime/E2E-validated` | `runtime-unverified`.
4. Final message: one status line + files changed.

**Never claim "verified", "working", or "passing" for anything you did not run.** You cannot run
gates, so almost everything you produce is `repository-derived`. Say so. An honest
`runtime-unverified` is worth more than a confident guess — the orchestrator is making sequencing
decisions from your labels.
