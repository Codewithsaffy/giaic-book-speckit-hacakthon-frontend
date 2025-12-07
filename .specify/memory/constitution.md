<!-- Sync Impact Report:
Version change: N/A → 1.0.0
Modified principles: None (new constitution)
Added sections: All principles from Article 1 to Article 7, plus Governance
Removed sections: None (completely new content)
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ updated
  - .specify/templates/spec-template.md: ✅ updated
  - .specify/templates/tasks-template.md: ✅ updated
  - .specify/templates/commands/*.md: ✅ updated
Templates requiring updates:
  - README.md: ⚠ pending
Follow-up TODOs: None
-->

# Book Writing Project Constitution

## Core Principles

### Content-First Philosophy
All work MUST focus exclusively on writing high-quality book content. No time shall be spent on validation, testing, or infrastructure changes unless explicitly blocking content creation.

### Bilingual Excellence
English content is written FIRST with technical accuracy and clarity. Roman Urdu translation follows immediately, maintaining technical terminology where appropriate. Both languages receive equal quality and attention to detail.

### Speed and Efficiency
Utilize subagents strategically to parallelize work. Write in focused sessions without interruption. Avoid perfectionism in first drafts - editing comes after completion. Complete modules sequentially to maintain momentum.

### Markdown as Source of Truth
All content MUST be in valid Markdown format. Follow Docusaurus conventions for structure and frontmatter. Use consistent heading hierarchy (# for titles, ## for sections, ### for subsections). Code blocks must specify language for syntax highlighting.

### Subagent Specialization
`md-writer` - Primary content creation agent. `content-architect` - Structure and outline planning. `book-editor` - Post-writing refinement and coherence. `research-gatherer` + `research-synthesizer` - Technical research only when needed.

### Technical Accuracy
Prioritize correctness in technical explanations. Use official documentation via Context7 MCP for accuracy. Include practical code examples for every concept. Reference real-world applications and case studies.

### Educational Structure
Each chapter follows: Introduction → Concepts → Examples → Hands-on → Summary. Progressive difficulty from fundamentals to advanced topics. Cross-reference related chapters using `crossref-manager`.

## Governance

All work must adhere to the core principles outlined above. Constitution supersedes all other practices. Amendments require documentation, approval, and migration plan if applicable. All PRs/reviews must verify compliance with these principles.

**Version**: 1.0.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-07