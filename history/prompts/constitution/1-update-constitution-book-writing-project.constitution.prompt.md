---
id: 1
title: update-constitution-book-writing-project
stage: constitution
date: 2025-12-07
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "governance", "book-writing"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
  - .specify/templates/plan-template.md
  - .specify/templates/spec-template.md
  - .specify/templates/tasks-template.md
tests: []
---

# PHR: Update Constitution - Book Writing Project

## Prompt Text

```
# Book Writing Project Constitution

## Article 1: Content-First Philosophy
All work MUST focus exclusively on writing high-quality book content. No time shall be spent on validation, testing, or infrastructure changes unless explicitly blocking content creation.

## Article 2: Bilingual Excellence
- English content is written FIRST with technical accuracy and clarity
- Roman Urdu translation follows immediately, maintaining technical terminology where appropriate
- Both languages receive equal quality and attention to detail

## Article 3: Speed and Efficiency
- Utilize subagents strategically to parallelize work
- Write in focused sessions without interruption
- Avoid perfectionism in first drafts - editing comes after completion
- Complete modules sequentially to maintain momentum

## Article 4: Markdown as Source of Truth
- All content MUST be in valid Markdown format
- Follow Docusaurus conventions for structure and frontmatter
- Use consistent heading hierarchy (# for titles, ## for sections, ### for subsections)
- Code blocks must specify language for syntax highlighting

## Article 5: Subagent Specialization
- `md-writer` - Primary content creation agent
- `content-architect` - Structure and outline planning
- `book-editor` - Post-writing refinement and coherence
- `research-gatherer` + `research-synthesizer` - Technical research only when needed

## Article 6: Technical Accuracy
- Prioritize correctness in technical explanations
- Use official documentation via Context7 MCP for accuracy
- Include practical code examples for every concept
- Reference real-world applications and case studies

## Article 7: Educational Structure
- Each chapter follows: Introduction → Concepts → Examples → Hands-on → Summary
- Progressive difficulty from fundamentals to advanced topics
- Cross-reference related chapters using `crossref-mana
```

## Response Text

Updated the constitution file and aligned templates to reflect the new book writing project principles, including Content-First Philosophy, Bilingual Excellence, Speed and Efficiency, Markdown as Source of Truth, Subagent Specialization, Technical Accuracy, and Educational Structure.

## Outcome

Successfully created a new constitution for the book writing project with 7 core principles and updated all dependent templates (plan, spec, tasks) to align with these principles. The constitution is now stored at .specify/memory/constitution.md with version 1.0.0.

## Evaluation

The constitution update was successful with all templates properly aligned to the new principles. The sync impact report was included in the constitution file, and all required fields in this PHR have been filled.