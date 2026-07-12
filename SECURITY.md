# Security Policy

## Supported versions

This project is pre-1.0. Security fixes land on `main`; please track the
latest commit.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's [private vulnerability reporting](https://github.com/ahmeddoghri/churnfm/security/advisories/new)
or email the maintainer. Include a description of the issue and its impact,
steps to reproduce (a minimal proof-of-concept helps), and any suggested
remediation.

You can expect an initial acknowledgement within a few days. Once a fix is
available it will be released and you will be credited unless you prefer to
remain anonymous.

## Scope notes

`churnfm` is a pure-stdlib library with no runtime dependencies and makes
no network calls on its own, so there's not much to attack out of the box.
When you plug in your own model, embedding, or storage backend, that
component's security posture is your responsibility. Validate and sanitize
any untrusted text you feed through the library, same as anywhere else.
