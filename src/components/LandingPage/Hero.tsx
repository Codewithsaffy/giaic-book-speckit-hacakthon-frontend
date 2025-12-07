import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function Hero() {
    return (
        <section className={clsx(styles.section, styles.heroSection)}>
            {/* Background Glow Orb */}
            <div className={styles.heroGlowOrb} />

            <div className={clsx(styles.container, styles.heroContainer)}>
                <div className={styles.heroContent}>
                    {/* Trust Badge / Eyebrow */}
                    <div className={clsx(styles.trustBadge, styles.animateFadeInUp)}>
                        <span className={styles.trustBadgeIcon}>🤖</span>
                        The Future of Robotics • ROS 2 & Isaac Sim • 2025 Edition
                    </div>

                    <h1 className={clsx(styles.heroTitle, styles.animateFadeInUp)} style={{ animationDelay: '0.1s' }}>
                        Physical AI & <br />
                        <span className={styles.highlightText}>Humanoid Robotics</span>
                    </h1>

                    <p className={clsx(styles.subheadline, styles.animateFadeInUp)} style={{ animationDelay: '0.2s' }}>
                        Master the complete stack: from building the nervous system with ROS 2
                        to training digital twins in NVIDIA Isaac and deploying Vision-Language-Action models.
                    </p>

                    <div className={clsx(styles.heroButtons, styles.animateFadeInUp)} style={{ animationDelay: '0.3s' }}>
                        <Link
                            className={styles.primaryButton}
                            to="/docs/intro">
                            Start Learning ROS 2
                        </Link>
                        <Link
                            className={clsx(styles.glassCard, styles.secondaryButton)}
                            to="/docs/category/module-1-ros2">
                            Explore Modules
                        </Link>
                    </div>
                </div>

                {/* Optional: Design Element / Abstract shape from reference */}
                <div className={styles.heroVisual}>
                    <div className={styles.codeWindowFrame}>
                        <div className={styles.codeControls}>
                            <span className={styles.controlDot} style={{ background: '#ff5f56' }}></span>
                            <span className={styles.controlDot} style={{ background: '#ffbd2e' }}></span>
                            <span className={styles.controlDot} style={{ background: '#27c93f' }}></span>
                        </div>
                        <pre className={styles.heroCodeBlock}>
                            <code>
                                <span style={{ color: '#ff79c6' }}>class</span> <span style={{ color: '#50fa7b' }}>HumanoidController</span>(<span style={{ color: '#8be9fd' }}>Node</span>):<br />
                                &nbsp;&nbsp;<span style={{ color: '#ff79c6' }}>def</span> <span style={{ color: '#f1fa8c' }}>__init__</span>(<span style={{ color: '#ff79c6' }}>self</span>):<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#ff79c6' }}>super</span>().<span style={{ color: '#f1fa8c' }}>__init__</span>(<span style={{ color: '#f1fa8c' }}>'brain_node'</span>)<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#ff79c6' }}>self</span>.joint_pub = <span style={{ color: '#ff79c6' }}>self</span>.create_publisher(<span style={{ color: '#8be9fd' }}>JointState</span>, <span style={{ color: '#f1fa8c' }}>'/cmd_vel'</span>, 10)<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#6272a4' }}># Initialize VLA Model connection...</span>
                            </code>
                        </pre>
                    </div>
                </div>
            </div>
        </section>
    );
}
