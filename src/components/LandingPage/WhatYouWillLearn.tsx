import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const features = [
    {
        title: 'ROS 2 Nervous System',
        desc: 'Deep dive into nodes, topics, actions, and services for robot control.',
        icon: '🧠'
    },
    {
        title: 'Digital Twins',
        desc: 'Simulate physics and environments with Gazebo Harmonics and Unity.',
        icon: '🏙️'
    },
    {
        title: 'NVIDIA Isaac Sim',
        desc: 'Leverage GPU-accelerated simulation and Isaac Lab for training agents.',
        icon: '⚡'
    },
    {
        title: 'VLA Models',
        desc: 'Implement Vision-Language-Action models for multimodal reasoning.',
        icon: '👁️'
    },
    {
        title: 'Humanoid Control',
        desc: 'Master kinematics, joint constraints, and URDF for humanoid robots.',
        icon: '🤖'
    },
    {
        title: 'Bilingual Learning',
        desc: 'Complete content available in both English and Roman Urdu.',
        icon: '🌐'
    },
];

export default function WhatYouWillLearn() {
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                <div className={styles.sectionHeader}>
                    <h2 className={styles.sectionTitle}>Master Physical AI</h2>
                    <p className={styles.subheadline}>
                        From low-level motor control to high-level reasoning with VLA models, get the complete skill set.
                    </p>
                </div>

                <div className={styles.learnGrid}>
                    {features.map((item, idx) => (
                        <div key={idx} className={clsx(styles.glassCard, styles.learnCard)}>
                            <div className={styles.iconBox}>{item.icon}</div>
                            <h3 className={styles.cardTitle}>{item.title}</h3>
                            <p className={styles.cardDesc}>{item.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
