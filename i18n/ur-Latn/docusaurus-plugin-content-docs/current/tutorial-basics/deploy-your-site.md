---
sidebar_position: 5
---

# Apni Site Deploy Karein

Docusaurus ek **static-site-generator** hai (jise **[Jamstack](https://jamstack.org/)** bhi kehte hain).

Yeh aap ki site ko simple **static HTML, JavaScript aur CSS files** ke tor par build karta hai.

## Apni Site Build Karein

Apni site ko **production ke liye** build karein:

```bash
npm run build
```

Static files `build` folder mein generate hoti hain.

## Apni Site Deploy Karein

Apni production build locally test karein:

```bash
npm run serve
```

`build` folder ab [http://localhost:3000/](http://localhost:3000/) par serve ho raha hai.

Ab aap `build` folder ko **taqreeban kahin bhi** asani se deploy kar sakte hain, **muft ya bahut kam kharche mein** (dekhen **[Deployment Guide](https://docusaurus.io/docs/deployment)**).
