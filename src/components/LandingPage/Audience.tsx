import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const audiences = [
    { label: 'Robotics Engineers', title: 'Professional Growth', desc: 'Transition from classical control to modern AI-driven robotics.' },
    { label: 'AI Researchers', title: 'Physical Intelligence', desc: 'Apply LLMs and VLA models to real-world embodiment.' },
    { label: 'Students', title: 'Structured Path', desc: 'A complete curriculum from "Hello World" to humanoid control.' },
    { label: 'Hobbyists', title: 'DIY Projects', desc: 'Build and simulate advanced robots without expensive hardware.' },
];

export default function Audience() {
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                <div className={styles.sectionHeader}>
                    <h2 className={styles.sectionTitle}>Built For Innovators</h2>
                </div>

                <div className={styles.audienceGrid}>
                    {audiences.map((aud, idx) => (
                        <div key={idx} className={clsx(styles.glassCard, styles.audienceCard)}>
                            <div className={styles.audienceLabel}>{aud.label}</div>
                            <h3 className={styles.audienceTitle}>{aud.title}</h3>
                            <p className={styles.audienceDesc}>{aud.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
