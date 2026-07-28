---
description: "Fetch all open Dependabot alerts, run Security Agent triage, cross-check with Security Validation Agent, and produce a dated report in docs/"
tools: ["read", "search", "web", "run_in_terminal"]
---

Run a full Dependabot security triage for the HASTE repo (microsoft/haste).

## Step 1 — Fetch open alerts

Run the following command to retrieve all open Dependabot alerts:

```bash
gh api repos/microsoft/haste/dependabot/alerts --paginate \
  -q '.[] | select(.state=="open") | {
    number: .number,
    severity: .security_vulnerability.severity,
    package: .security_vulnerability.package.name,
    ecosystem: .security_vulnerability.package.ecosystem,
    cve: .security_advisory.cve_id,
    summary: .security_advisory.summary,
    patched: .security_vulnerability.first_patched_version.identifier,
    vulnerable: .security_vulnerability.vulnerable_version_range,
    manifest: .dependency.manifest_path
  }'
```

If there are no open alerts, output a brief report stating that and stop.

## Step 2 — Security Agent triage

Using the agent definition in `.github/agents/security.agent.md` and the skill in `.github/skills/security-analysis/SKILL.md`, perform a full triage of the alerts retrieved in Step 1.

For each alert:
- Group alerts by package (multiple CVEs for the same package → one group)
- Assess HASTE-specific impact using context from `.github/copilot-instructions.md`:
  - Is the vulnerable code path reachable in production?
  - Is the package in the browser bundle, a Node.js dev tool, or a Python pipeline container?
  - What data is at risk? What is the blast radius?
- Apply severity SLAs: Critical → same day, High → 3 business days, Medium → next sprint, Low → backlog
- Produce a recommended action for each group (upgrade, override, accept risk)
- Note any packages where a major version jump requires additional testing

## Step 3 — Security Validation Agent cross-check

Using the agent definition in `.github/agents/security-validation.agent.md`, validate every finding from Step 2.

For each finding, follow the checklist in the agent definition exactly:
- Search **both** NVD (`nvd.nist.gov`) and GitHub Advisory Database (`github.com/advisories`) independently
- Never declare a CVE unverifiable after checking only one source
- Read the **full affected version table** for all maintained branches in the advisory — do not rely on the summary description alone
- Confirm patched versions exist on npm or PyPI
- Verify HASTE's pinned version falls within the affected range
- Check that production impact classification is correct

Flag each finding as: ✅ Validated | ⚠️ Corrected | ❌ Rejected

## Step 4 — Produce the report

Today's date: use the current date in YYYY-MM-DD format.

Create a new file at `docs/security-triage-YYYY-MM-DD.md` using the structure from `docs/security-triage-2026-06-23.md` as a template.

The report must include:
- Header with repo, alert count, generation date, status
- Executive summary table with severity, group, alert count, production impact, status
- Per-finding sections with: Severity, Package, CVE, Manifest, Component, Status, description, HASTE impact assessment, recommended fix, references
- Validation Notes section distinguishing confirmed errors from accurate corrections
- Applied Remediations section (leave as "pending human approval" until changes are made)

**Status line for all findings:** `⏳ Pending human approval — no code changes made`

## Output

Confirm the file was created at `docs/security-triage-YYYY-MM-DD.md` and summarise:
- Total alerts found
- How many validated / corrected / rejected by the Validation Agent
- The single highest-priority action required
