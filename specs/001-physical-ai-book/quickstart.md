# Quickstart: Physical AI & Humanoid Robotics Book

## Getting Started

This guide will help you get started with the Physical AI & Humanoid Robotics Book project.

### Prerequisites

- Node.js (version 18 or higher)
- npm or yarn package manager
- Git for version control
- Basic knowledge of Markdown and Docusaurus

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd docusurus-frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**:
   ```bash
   npm start
   # or
   yarn start
   ```

4. **Open your browser** to `http://localhost:3000` to view the book.

### Content Structure

The book is organized into 4 modules:

1. **Module 1: The Robotic Nervous System (ROS 2)** - Covers ROS 2 fundamentals
2. **Module 2: The Digital Twin (Gazebo & Unity)** - Simulation environments
3. **Module 3: NVIDIA Isaac Platform** - Advanced robotics platform
4. **Module 4: Vision-Language-Action Systems** - AI integration

### Creating Content

1. **Add new sections** to existing chapters in the `docs/` directory
2. **Follow the frontmatter standard**:
   ```yaml
   ---
   title: "Section Title"
   description: "Brief 1-2 sentence description"
   sidebar_position: N
   keywords: [keyword1, keyword2, keyword3]
   ---
   ```

3. **Use proper Markdown formatting** with appropriate heading hierarchy
4. **Include code examples** with language specification for syntax highlighting

### Bilingual Content

1. **Create English content first** in the `docs/` directory
2. **Add Roman Urdu translation** in the `i18n/ur-Latn/docusaurus-plugin-content-docs/current/` directory
3. **Maintain technical terminology consistency** between both versions

### Running the Site

- **Development**: `npm start` - Starts a local development server with hot reloading
- **Build**: `npm run build` - Creates a production build in the `build/` directory
- **Serve**: `npm run serve` - Locally serve the production build
- **Deploy**: `npm run deploy` - Deploy to GitHub Pages (if configured)

### Key Directories

- `docs/` - English content organized by modules and chapters
- `i18n/ur-Latn/` - Roman Urdu translations
- `specs/001-physical-ai-book/` - Project specifications and plans
- `src/` - Custom React components and site configuration