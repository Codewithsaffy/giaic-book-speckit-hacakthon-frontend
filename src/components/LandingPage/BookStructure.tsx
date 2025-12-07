import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const chapters = [
    { num: '01', title: 'The Robotic Nervous System', desc: 'ROS 2, Nodes, Topics, Services, Actions, Parameters' },
    { num: '02', title: 'The Digital Twin', desc: 'Gazebo Harmonics, Unity Integration, Physics Simulation' },
    { num: '03', title: 'NVIDIA Isaac Platform', desc: 'Isaac Sim, Isaac Lab, GPU Acceleration, Reinforcement Learning' },
    { num: '04', title: 'Vision-Language-Action', desc: 'Multimodal Models, VLA Architecture, Real-time Inference' },
    { num: '05', title: 'Humanoid Control', desc: 'Kinematics, Dynamics, URDF, Whole-Body Control' },
    { num: '06', title: 'Deployment', desc: 'Real-world Transfer, Latency Optimization, Safety' },
];

export default function BookStructure() {
    return (
        <section className={clsx(styles.section, styles.structureSection)}>
            <div className={styles.container}>
                <div className={styles.splitLayout}>
                    <div className={styles.splitContent}>
                        <h2 className={styles.sectionTitle}>Structured for Mastery</h2>
                        <p className={styles.subheadline} style={{ margin: 0 }}>
                            From core ROS 2 concepts to advanced Physical AI implementations.
                            4 Modules, 40+ Chapters.
                        </p>
                        <div className={styles.chapterList}>
                            {chapters.map((chapter, idx) => (
                                <div key={idx} className={styles.chapterItem}>
                                    <span className={styles.chapterNum}>{chapter.num}</span>
                                    <div>
                                        <h4 className={styles.chapterTitle}>{chapter.title}</h4>
                                        <span className={styles.chapterDesc}>{chapter.desc}</span>
                                    </div>
                                </div>
                            ))}
                            <div className={styles.moreChapters}>
                                + Hands-on Code Labs in Every Chapter
                            </div>
                        </div>
                        <Link className={clsx(styles.glassCard, styles.secondaryButton)} to="/docs/intro">
                            View Full Table of Contents →
                        </Link>
                    </div>

                    <div className={styles.splitVisual}>
                        {/* Abstract Book/Structure Visual */}
                        <div className={styles.bookVisualCard}>
                            <div className={styles.bookVisualSpine}></div>
                            <div className={styles.bookVisualCover}>
                                <div className={styles.bookVisualLogo}>AI</div>
                                <div className={styles.bookVisualTitle}>Physical<br />AI Book</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
