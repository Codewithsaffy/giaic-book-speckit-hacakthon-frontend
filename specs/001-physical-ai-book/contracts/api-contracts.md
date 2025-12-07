# API Contracts: Physical AI & Humanoid Robotics Book

## Cross-Reference API

### Get Section by ID
```
GET /api/sections/{sectionId}
```

**Description**: Retrieve a specific section by its ID to enable cross-referencing between different parts of the book.

**Parameters**:
- sectionId (path): Unique identifier for the section (required)

**Response**:
```
{
  "id": "string",
  "title": "string",
  "content_preview": "string",
  "module": "string",
  "chapter": "string",
  "position": "integer",
  "related_sections": ["string"]
}
```

### Search Sections
```
GET /api/sections/search
```

**Description**: Search for sections containing specific terms to enable internal linking and navigation.

**Parameters**:
- query (query): Search term (required)
- limit (query): Maximum number of results (default: 10)

**Response**:
```
{
  "results": [
    {
      "id": "string",
      "title": "string",
      "module": "string",
      "chapter": "string",
      "relevance_score": "number"
    }
  ]
}
```

## Translation API

### Get Bilingual Pair
```
GET /api/bilingual/{contentId}
```

**Description**: Retrieve both English and Roman Urdu versions of a content piece.

**Parameters**:
- contentId (path): ID of the English content (required)

**Response**:
```
{
  "english_content": {
    "id": "string",
    "title": "string",
    "content": "string"
  },
  "roman_urdu_content": {
    "id": "string",
    "title": "string",
    "content": "string"
  },
  "technical_terms": {
    "key": "string"
  }
}
```