---
name: docusaurus-validator
description: Tests Docusaurus builds, validates internal/external links, checks search indexing, and ensures production-readiness. Use before committing chapters or deploying updates.
allowed-tools: Read, Grep, Glob, Bash
---

# Docusaurus Build Validator

## Purpose
Catch Docusaurus build errors, broken links, and configuration issues before they reach production.

## Pre-Build Validation
1. **Configuration check**: Verify `docusaurus.config.js` has all required plugins
2. **Sidebar integrity**: Ensure `sidebars.js` references all created docs
3. **Static assets**: Check all images in `static/img/` are referenced correctly
4. **Plugin requirements**: Confirm required npm packages are installed

## Build Testing

Run full build process and check for issues:

```bash
npm run build
```

Verify:
- No build errors
- No "broken links" warnings
- All pages are generated

## Post-Build Checks
1. **Link validation**: Test all internal links in `build/` directory
2. **Search indexing**: Verify Algolia config if enabled
3. **Performance**: Run Lighthouse CI on generated files
4. **Accessibility**: Check axe-core compliance

## Link Integrity
- **Internal links**: All `[text](./file.md)` must resolve
- **Heading anchors**: Verify `#section-name` anchors exist
- **External links**: Test with `curl` or fetch tool (optional, may be slow)

## Output
Create `validation-reports/{timestamp}-build-report.md` with:
- Build success/failure status
- Broken links list with suggested fixes
- Missing asset warnings
- Performance scores
- Recommendations for fixes

## Invocation
Activates when you: "Validate the build" or "Check if everything compiles" or "Test before deploy"