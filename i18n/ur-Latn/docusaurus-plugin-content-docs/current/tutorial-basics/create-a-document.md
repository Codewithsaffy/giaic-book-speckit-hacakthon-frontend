---
sidebar_position: 2
---

# Ek Document Banayein

Documents **pages ke groups** hain jo in cheezon se connected hain:

- ek **sidebar**
- **previous/next navigation**
- **versioning**

## Apna Pehla Doc Banayein

`docs/hello.md` par ek Markdown file banayein:

```md title="docs/hello.md"
# Hello

Yeh mera **pehla Docusaurus document** hai!
```

Ab ek naya document yahan available hai: [http://localhost:3000/docs/hello](http://localhost:3000/docs/hello)

## Sidebar Configure Karein

Docusaurus automatically `docs` folder se ek **sidebar banata hai**.

Sidebar label aur position customize karne ke liye metadata add karein:

```md title="docs/hello.md" {1-4}
---
sidebar_label: 'Salaam!'
sidebar_position: 3
---

# Hello

Yeh mera **pehla Docusaurus document** hai!
```

Aap apna sidebar explicitly `sidebars.js` mein bhi bana sakte hain:

```js title="sidebars.js"
export default {
  tutorialSidebar: [
    'intro',
    // highlight-next-line
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
};
```
