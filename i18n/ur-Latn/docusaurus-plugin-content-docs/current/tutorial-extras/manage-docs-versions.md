---
sidebar_position: 1
---

# Docs Versions Manage Karein

Docusaurus aap ke docs ke multiple versions manage kar sakta hai.

## Ek Docs Version Banayein

Apne project ka version 1.0 release karein:

```bash
npm run docusaurus docs:version 1.0
```

`docs` folder `versioned_docs/version-1.0` mein copy ho jata hai aur `versions.json` ban jata hai.

Ab aap ke docs ke 2 versions hain:

- `1.0` jo `http://localhost:3000/docs/` par hai version 1.0 docs ke liye
- `current` jo `http://localhost:3000/docs/next/` par hai **aane wale, unreleased docs** ke liye

## Version Dropdown Add Karein

Versions ke beech asani se navigate karne ke liye, ek version dropdown add karein.

`docusaurus.config.js` file ko modify karein:

```js title="docusaurus.config.js"
export default {
  themeConfig: {
    navbar: {
      items: [
        // highlight-start
        {
          type: 'docsVersionDropdown',
        },
        // highlight-end
      ],
    },
  },
};
```

Docs version dropdown aap ke navbar mein appear hota hai:

![Docs Version Dropdown](./img/docsVersionDropdown.png)

## Ek Existing Version Update Karein

Versioned docs ko unke respective folder mein edit karna mumkin hai:

- `versioned_docs/version-1.0/hello.md` update karta hai `http://localhost:3000/docs/hello`
- `docs/hello.md` update karta hai `http://localhost:3000/docs/next/hello`
