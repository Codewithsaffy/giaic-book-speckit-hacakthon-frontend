---
name: markdown-linter
description: Validates Markdown syntax, enforces book style guidelines, checks heading hierarchy, and ensures consistency across all chapters. Use after writing or editing any markdown content.
allowed-tools: Read, Write, Edit, Grep, Glob
---

# Markdown Linter & Style Enforcer

## Purpose
Maintain consistent style, structure, and formatting across all book chapters using your established constitution.

## Linting Rules

### Syntax Validation
- **No MDX/React syntax**: Flag any JSX, React components, or custom HTML
- **Proper code blocks**: Verify language identifiers (```javascript, ```bash, etc.)
- **Valid Mermaid syntax**: Check diagram syntax if mermaid code blocks exist
- **Tab component integrity**: Ensure `<Tabs>` and `<TabItem>` are properly closed

### Style Enforcement (from Constitution)
- **Reading time**: Each section should be 8-12 minutes (target: ~1,600-2,400 words)
- **Exercise frequency**: At least 3 practical exercises per major section
- **Heading hierarchy**: H2 for main sections, H3 for subsections (no skipping levels)
- **Cross-references**: Verify relative links: `[text](./other-file.md#heading)`

### Content Structure
- **Frontmatter validation**: Check required fields (title, description, keywords, sidebar_position)
- **Admonition usage**: Ensure proper admonition syntax: `:::note`, `:::warning`, etc.
- **Image references**: Verify relative paths to `static/img/` and alt text presence

## Automated Fixes
- Fix inconsistent heading levels
- Add missing language identifiers to code blocks
- Standardize admonition formatting
- Correct cross-reference syntax

## Output
Create `lint-reports/{chapter}-lint.md` with:
- **Critical errors**: Must fix before publication
- **Warnings**: Should fix for consistency
- **Suggestions**: Optional improvements

## Invocation
Automatically runs when you say: "Lint this chapter" or "Check style consistency" or after any Edit tool modifies markdown files.