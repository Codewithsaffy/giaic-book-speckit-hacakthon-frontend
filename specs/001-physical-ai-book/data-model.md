# Data Model: Physical AI & Humanoid Robotics Book

## Entity: Book Module
- **Fields**:
  - id: string (unique identifier, e.g., "module-1-ros2")
  - title: string (display title)
  - description: string (brief description)
  - position: integer (order in the book sequence)
  - chapters: array of Chapter entities
- **Validation rules**:
  - id must be unique across all modules
  - position must be positive integer
  - title and description are required
- **Relationships**: Contains multiple Chapter entities

## Entity: Chapter
- **Fields**:
  - id: string (unique identifier, e.g., "chapter-1-introduction-to-ros2")
  - title: string (display title)
  - description: string (brief description)
  - position: integer (order within parent module)
  - sections: array of Section entities
  - module_id: string (reference to parent Module)
- **Validation rules**:
  - id must be unique across all chapters
  - position must be positive integer
  - title and description are required
  - must belong to exactly one module
- **Relationships**: Belongs to one Module, contains multiple Section entities

## Entity: Section
- **Fields**:
  - id: string (unique identifier, e.g., "what-is-ros2")
  - title: string (display title)
  - content: string (Markdown content)
  - position: integer (order within parent chapter)
  - frontmatter: object (title, description, sidebar_position, keywords)
  - chapter_id: string (reference to parent Chapter)
  - cross_references: array of string (IDs of related sections)
- **Validation rules**:
  - id must be unique across all sections
  - position must be positive integer
  - title and content are required
  - must belong to exactly one chapter
  - frontmatter must follow Docusaurus conventions
- **Relationships**: Belongs to one Chapter, references multiple other Section entities

## Entity: Bilingual Content Pair
- **Fields**:
  - english_content_id: string (reference to English Section)
  - roman_urdu_content_id: string (reference to Roman Urdu Section)
  - content_type: enum ("section", "chapter", "module")
  - technical_terms: object (mapping of English to Roman Urdu technical terms)
- **Validation rules**:
  - Both content references must exist
  - content_type must be one of allowed values
  - technical_terms must maintain consistency across the book
- **Relationships**: References two Section entities (one English, one Roman Urdu)

## Entity: Code Example
- **Fields**:
  - id: string (unique identifier)
  - title: string (description of the example)
  - code: string (actual code content)
  - language: string (programming language for syntax highlighting)
  - description: string (explanation of what the code does)
  - section_id: string (reference to parent Section)
  - simulation_environment: string (where the example can be tested)
- **Validation rules**:
  - id must be unique
  - code and language are required
  - must belong to exactly one section
- **Relationships**: Belongs to one Section entity

## State Transitions
- **Content Creation**: Draft → Review → Approved → Published
- **Translation Status**: English Only → Translating → Roman Urdu Ready → Verified
- **Quality Status**: Incomplete → Complete → Validated → Published