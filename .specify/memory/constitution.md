<!--
Sync Impact Report:
- Version change: N/A → 1.0.0 (initial version)
- List of modified principles: N/A (new constitution)
- Added sections: Core Principles, Non-Negotiable Rules, Quality Standards, Governance
- Removed sections: N/A
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/templates/commands/*.md ⚠ pending
- Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Book Constitution

## Core Principles

### Bilingual Excellence
Every concept must be accurately presented in both English and Roman Urdu with equal technical depth

### Progressive Complexity
Content must build from fundamentals to advanced topics within each module

### Practical Relevance
Every theoretical concept must include real-world applications and code examples

### Agent-Driven Quality
All content must pass through Claude Code validation agents before finalization

## Non-Negotiable Rules

1. **Zero Configuration Changes**: No modifications to existing Docusaurus/i18n infrastructure
2. **Atomic Chapter Creation**: Each chapter must be completed and validated before proceeding to the next
3. **Cross-Language Consistency**: Technical terms must maintain identical meaning across both languages
4. **Agent Workflow Compliance**: All content must follow the specified Claude Code agent workflow
5. **Validation Gates**: No content progresses without passing all quality validation steps

## Quality Standards

- **Technical Accuracy**: All code examples and concepts must be validated against current ROS 2/Isaac documentation
- **Readability**: Maximum 300 words per paragraph with clear section breaks
- **Visual Balance**: Minimum 1 diagram per 500 words of technical content
- **Audience Alignment**: Content depth must match graduate-level engineering expectations

## Governance

- **speckit-orchestrator** manages workflow compliance
- **book-editor** agent performs final quality assurance
- **Weekly validation cycles** ensure consistency across all modules

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06
