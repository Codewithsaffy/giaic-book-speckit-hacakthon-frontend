---
name: crossref-manager
description: Manages cross-references between chapters, maintains glossary consistency, updates links when files move, and ensures citation integrity across the entire book. Use when re-organizing content or after major edits.
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

# Cross-Reference & Citation Manager

## Purpose
Maintain link integrity and terminological consistency across all book chapters as the project evolves.

## Core Functions

### 1. Link Inventory
Generate a complete map of all internal links:
```bash
# Find all markdown links
grep -r "\[.*\]\(\.*/.*\)" docs/
```

### 2. Glossary Management
- **Term extraction**: Identify and extract key terms from all chapters
- **Definition consistency**: Ensure terms are defined consistently
- **Auto-linking**: Suggest glossary links for first use of terms
- **Glossary file**: Maintain `docs/glossary.md` with alphabetical terms

### 3. Reference Validation
- **Broken link detection**: Find dead internal links
- **Outdated citations**: Flag references to old research
- **Version drift**: Check code examples match current versions

### 4. Reorganization Support
When moving files:
- **Link updates**: Automatically update all references to moved files
- **Redirect mapping**: Create redirect rules for changed URLs
- **Sidebar updates**: Sync `sidebars.js` with new structure

## Output Files
1. **Link map**: `book-metadata/link-map.md` (all internal connections)
2. **Glossary**: `docs/glossary.md` (maintained automatically)
3. **Broken links**: `book-metadata/broken-links.md` (requires manual fix)

## Integration
Works with `docusaurus-validator` to ensure link integrity before builds.

## Invocation
Activates when: "Update links after moving files" or "Check cross-references" or "Sync glossary"