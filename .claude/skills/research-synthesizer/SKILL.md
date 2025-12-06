---
name: research-synthesizer
description: Automatically synthesizes research findings from Tavily and Context7 into structured briefs for book chapters. Use when research is complete and you need to organize findings into a usable format for writing.
allowed-tools: Read, Write, Edit, Grep, Glob
---

# Research Synthesizer

## Purpose
Transform raw research data from Tavily and Context7 into structured, citation-rich research briefs that feed directly into your writing process.

## Workflow
1. **Gather Sources**: Find all research files in `research/{topic}/` using Glob
2. **Extract Key Findings**: Use Grep to identify statistics, quotes, and key points
3. **Synthesize Structure**: Create a comprehensive brief with:
   - Executive summary (3-5 sentences)
   - Key statistics with confidence ratings
   - Expert quotes with proper attribution
   - Code examples grouped by complexity
   - Gap analysis (what's missing from literature)
   - Suggested chapter outline based on research
4. **Citations**: Format all sources with URLs, access dates, and credibility scores

## Output Format
Create `research/{topic}/synthesized-brief.md` with clear sections:
- `## Key Findings`
- `## Supporting Evidence`
- `## Code Examples`
- `## Literature Gaps`
- `## Recommended Chapter Structure`

## Example Usage
When you have raw research files, simply say: "Synthesize the research for chapter 3" and this skill will automatically locate and compile everything.