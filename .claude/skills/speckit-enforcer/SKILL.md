---
name: speckit-enforcer
description: Enforces Speckit specification-driven development workflow for book chapters. Validates that constitution, specifications, plans, and tasks exist before allowing writing to proceed. Use when starting a new chapter or verifying workflow compliance.
allowed-tools: Read, Grep, Glob, Bash
---

# Speckit Workflow Enforcer

## Purpose
Ensure no chapter writing begins without proper Speckit workflow compliance, preventing "vibe writing" and ensuring quality through structured checkpoints.

## Validation Checklist
Before allowing content creation, verify:

### Phase 1: Foundation Files
- [ ] `.claude/constitution.md` exists and is current
- [ ] `chapters/{chapter-name}/constitution.md` exists (chapter-specific principles)
- [ ] `chapters/{chapter-name}/spec.md` exists with clear learning objectives

### Phase 2: Planning Files  
- [ ] `chapters/{chapter-name}/plan.md` exists with technical decisions
- [ ] Plan includes markdown extension requirements
- [ ] Plan specifies code example architecture
- [ ] Plan identifies required Docusaurus plugins

### Phase 3: Task Breakdown
- [ ] `chapters/{chapter-name}/tasks.md` exists with atomic tasks (<500 words each)
- [ ] Tasks are sequenced properly with dependencies marked
- [ ] Parallel execution markers exist where appropriate

## Enforcement Actions
If any file is missing or incomplete:
1. **Block writing** until requirements are met
2. **Generate placeholder files** with templates
3. **Alert with specific missing elements**
4. **Suggest next steps** to complete workflow

## Automatic Invocation
This skill activates whenever you request: "Start writing chapter X" or "Create content for section Y" - ensuring workflow compliance before any writing begins.