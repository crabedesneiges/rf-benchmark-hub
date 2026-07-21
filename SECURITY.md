# Security Policy

RF-Benchmark-Hub is a benchmark and leaderboard repository. It ships **code**
(the `rfbench` package, the CLI, the site generator), **JSON schemas**, and
**split indices + checksums** — it deliberately ships **no raw datasets, no model
weights, and no secrets** (see the "no raw data" rule in `CONTRIBUTING`/`CLAUDE.md`
and the CI check `tools/check_no_raw_data.py`). Please keep that in mind when
assessing impact: there is no production service and no credential store here.

## What counts as a security issue

We are interested in reports such as:

* **Code-execution / supply-chain risks** — e.g. a `data prepare`/`data download`
  path, a checkpoint loader, or a deserialization step (`torch.load`, `pickle`,
  YAML, JSON) that could execute arbitrary code from an attacker-controlled file
  or URL.
* **Path traversal / arbitrary write** — a download or extraction routine that
  could write outside `RFBENCH_CACHE`.
* **Leaked secrets or raw data** — a credential, token, private URL, or a
  committed dataset/weights file that slipped past the `no-raw-data` CI gate.
* **Vulnerable or malicious dependency** — a pinned dependency with a known CVE
  that is reachable from our code paths.

Ordinary bugs, incorrect leaderboard numbers, reproduction mismatches, or
feature requests are **not** security issues — please use the regular
[issue templates](.github/ISSUE_TEMPLATE/) instead.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** so we can address them
before they are widely known. Preferred channels, in order:

1. **GitHub private vulnerability reporting** — open a report via the repository's
   *Security* tab → *Report a vulnerability* (GitHub Security Advisories). This
   keeps the discussion private with the maintainers and is the preferred channel.
2. **Private message to a maintainer** — if private reporting is not yet enabled on
   the repository, contact a maintainer privately through their GitHub profile and
   ask them to open a private advisory. Do **not** put vulnerability details in a
   public issue; a short public note asking a maintainer to enable private reporting
   is acceptable only if you have no other channel.

Please **do not** open a public issue or pull request for a vulnerability until
a fix (or a mitigation) has been agreed with the maintainers.

Include, as much as you can:

* the affected file(s)/command(s) and the commit or release,
* a description of the impact and a minimal way to reproduce it,
* any suggested fix or mitigation.

## Our commitment

* We aim to **acknowledge** a report within **7 days**.
* We will keep you informed of our assessment and of the fix timeline.
* With your permission, we will **credit** you in the release notes once a fix
  ships.

Thank you for helping keep RF-Benchmark-Hub and its users safe.
