import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

export default function KeyFeatures() {
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                <div className={styles.sectionHeader}>
                    <h2 className={styles.sectionTitle}>Why This Book?</h2>
                </div>

                <div className={styles.bentoGrid}>
                    {/* Large Card 1 - Simulation First - Area: sim */}
                    <div className={clsx(styles.glassCard, styles.bentoCard, styles.featureSim)}>
                        <div className={styles.bentoContent}>
                            <h3>Simulation-First Approach</h3>
                            <p>Don't break expensive hardware. Master concepts in Gazebo Harmonics and Isaac Sim before deployment.</p>
                            <div className={styles.bentoVisualGradient}></div>
                        </div>
                    </div>

                    {/* Small Card 1 - Production ROS 2 - Top Right - Area: ros */}
                    <div className={clsx(styles.glassCard, styles.bentoCard, styles.featureRos)}>
                        <h3>Production ROS 2</h3>
                        <p>Best practices for nodes & lifecycle.</p>
                    </div>

                    {/* Small Card 2 - Bilingual - Bottom Right (of top section) - Area: bi */}
                    <div className={clsx(styles.glassCard, styles.bentoCard, styles.featureBi)}>
                        <h3>Bilingual Support</h3>
                        <p>English & Roman Urdu.</p>
                    </div>

                    {/* Wide Card - Physical AI - Bottom Right (wide) - Area: phy */}
                    <div className={clsx(styles.glassCard, styles.bentoCard, styles.featurePhy)}>
                        <div className={styles.bentoContent}>
                            <h3>Physical AI Integration</h3>
                            <p>Bridge the gap between Large Language Models and physical actuation with VLA.</p>
                        </div>
                    </div>

                    {/* Tall/Standard Card - Open Source - Bottom Left - Area: open */}
                    <div className={clsx(styles.glassCard, styles.bentoCard, styles.featureOpen)}>
                        <div className={styles.bentoContent}>
                            <h3>Open Source</h3>
                            <p>100% Free & Open.</p>
                            <div className={styles.bentoIconBig}>📖</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
