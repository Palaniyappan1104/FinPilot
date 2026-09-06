# Contributing to FinPilot

Thank you for contributing to FinPilot! This guide outlines our branching strategy, commit conventions, coding guidelines, and security requirements.

---

## Branching Strategy

- **`main`**: The primary branch containing tested, phase-verified code.
- **Phase / Feature Branches**: Work on specific phases or features should be done in dedicated branches before merging into `main`:
  - `phase/xx-phase-name` (e.g., `phase/01-foundation`, `phase/02-agent-infra`)
  - `feature/feature-name`
  - `fix/bug-description`

---

## Commit Message Conventions

We follow a structured commit convention:

- **Phase Checkpoint Commits**:
  - `Phase X: <Phase Name>` (e.g., `Phase 0: Project Planning & Repository Setup`)
- **Intermediate Commits**:
  - `feat: <description>` — New feature or capability
  - `fix: <description>` — Bug fix
  - `docs: <description>` — Documentation changes
  - `refactor: <description>` — Code refactoring without behavioral change
  - `test: <description>` — Adding or updating tests
  - `chore: <description>` — Build, tooling, or repository maintenance

---

## Development & Code Style

- Refer to `.editorconfig` for formatting rules:
  - **Python**: 4 spaces indentation, UTF-8, LF line endings.
  - **TypeScript / React / JSON / Markdown**: 2 spaces indentation, UTF-8, LF line endings.
- Keep dependencies lean and justified.

---

## Security & Secrets

- **NEVER** commit API keys, passwords, secret tokens, private certificates, or environment files (`.env`).
- Always use `.env.example` templates with sanitized placeholder values.
- Check `git status` before committing to avoid staging unintended files.

---

## Roadmap Alignment

All development work is coordinated according to the master roadmap in [plan.md](plan.md). Ensure tasks map directly to the active phase deliverables.
