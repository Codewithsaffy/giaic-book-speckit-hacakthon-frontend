---
sidebar_position: 2
---

# Apni Site Translate Karein

Chalein `docs/intro.md` ko French mein translate karein.

## i18n Configure Karein

`docusaurus.config.js` ko modify karein `fr` locale ke liye support add karne ke liye:

```js title="docusaurus.config.js"
export default {
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'fr'],
  },
};
```

## Ek Doc Translate Karein

`docs/intro.md` file ko `i18n/fr` folder mein copy karein:

```bash
mkdir -p i18n/fr/docusaurus-plugin-content-docs/current/

cp docs/intro.md i18n/fr/docusaurus-plugin-content-docs/current/intro.md
```

`i18n/fr/docusaurus-plugin-content-docs/current/intro.md` ko French mein translate karein.

## Apni Localized Site Start Karein

Apni site French locale par start karein:

```bash
npm run start -- --locale fr
```

Aap ki localized site [http://localhost:3000/fr/](http://localhost:3000/fr/) par accessible hai aur `Getting Started` page translate ho gaya hai.

:::caution

Development mein, aap ek waqt mein sirf ek locale use kar sakte hain.

:::

## Ek Locale Dropdown Add Karein

Languages ke beech asani se navigate karne ke liye, ek locale dropdown add karein.

`docusaurus.config.js` file ko modify karein:

```js title="docusaurus.config.js"
export default {
  themeConfig: {
    navbar: {
      items: [
        // highlight-start
        {
          type: 'localeDropdown',
        },
        // highlight-end
      ],
    },
  },
};
```

Locale dropdown ab aap ke navbar mein appear hota hai:

![Locale Dropdown](./img/localeDropdown.png)

## Apni Localized Site Build Karein

Ek specific locale ke liye apni site build karein:

```bash
npm run build -- --locale fr
```

Ya apni site ko sab locales ke saath ek saath build karein:

```bash
npm run build
```
