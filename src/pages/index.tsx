import type { ReactNode } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import {
  Hero,
  QuickValue,
  WhatYouWillLearn,
  BookStructure,
  InteractivePreview,
  Audience,
  KeyFeatures,
  SocialProof,
  GettingStarted,
  FAQ,
  Community,
  RelatedResources,
  FinalCTA
} from '@site/src/components/LandingPage';
import styles from '@site/src/components/LandingPage/styles.module.css';

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="Master Modern Web Development with our physical AI book.">
      <main className={styles.landingPage}>
        <Hero />
        <QuickValue />
        <WhatYouWillLearn />
        <BookStructure />
        <InteractivePreview />
        <Audience />
        <KeyFeatures />
        <SocialProof />
        <GettingStarted />
        <FAQ />
        <Community />
        <RelatedResources />
        <FinalCTA />
      </main>
    </Layout>
  );
}
