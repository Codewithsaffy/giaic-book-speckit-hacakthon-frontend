---
sidebar_position: 1
---

# Ek Page Banayein

**Markdown ya React** files ko `src/pages` mein add karein aur ek **standalone page** banayein:

- `src/pages/index.js` → `localhost:3000/`
- `src/pages/foo.md` → `localhost:3000/foo`
- `src/pages/foo/bar.js` → `localhost:3000/foo/bar`

## Apna Pehla React Page Banayein

`src/pages/my-react-page.js` par ek file banayein:

```jsx title="src/pages/my-react-page.js"
import React from 'react';
import Layout from '@theme/Layout';

export default function MyReactPage() {
  return (
    <Layout>
      <h1>Mera React page</h1>
      <p>Yeh ek React page hai</p>
    </Layout>
  );
}
```

Ab ek naya page yahan available hai: [http://localhost:3000/my-react-page](http://localhost:3000/my-react-page)

## Apna Pehla Markdown Page Banayein

`src/pages/my-markdown-page.md` par ek file banayein:

```mdx title="src/pages/my-markdown-page.md"
# Mera Markdown page

Yeh ek Markdown page hai
```

Ab ek naya page yahan available hai: [http://localhost:3000/my-markdown-page](http://localhost:3000/my-markdown-page)
