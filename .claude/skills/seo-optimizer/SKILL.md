---
name: seo-optimizer
description: Optimizes book content for search engines and Docusaurus discovery. Enhances frontmatter, suggests keywords, improves descriptions, and structures content for better indexing. Use after writing but before editing.
allowed-tools: Read, Write, Edit, Grep
---

# SEO & Discovery Optimizer

## Purpose
Maximize your book's discoverability through search engines and Docusaurus' built-in search.

## Frontmatter Enhancement
### Required Fields
- `title`: 50-60 characters, keyword-rich, compelling
- `description`: 150-160 characters, clear value proposition
- `keywords`: 5-10 relevant terms, comma-separated
- `sidebar_position`: Logical ordering number
- `slug`: (Optional) Custom URL path

### SEO Analysis
1. **Keyword research**: Use Tavily to find trending terms in your topic
2. **Competitor analysis**: Identify high-ranking similar content
3. **Search intent matching**: Align content with user queries

## Content Optimization
- **H1 heading**: Must include primary keyword (chapter title)
- **First paragraph**: Clear summary with main keywords
- **Heading distribution**: Keywords naturally in H2/H3 headings
- **Image alt text**: Descriptive, keyword-inclusive
- **Internal linking**: 3-5 links to related chapters

## Technical SEO
- **URL structure**: Verify slug is clean and readable
- **Meta tags**: Check Open Graph and Twitter card tags
- **Schema markup**: Suggest structured data for tutorials/guides
- **Mobile optimization**: Ensure responsive design compliance

## Docusaurus-Specific
- **Algolia config**: Suggest searchable synonyms and attributes
- **Search keywords**: Add hidden search terms in frontmatter
- **Doc tags**: Enable and configure for content grouping

## Output
Edit the markdown file directly to add optimized frontmatter and suggest content improvements.

## Invocation
Triggers when you say: "Optimize this for SEO" or "Improve discoverability" or "Add search keywords"