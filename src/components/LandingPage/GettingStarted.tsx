import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import cls from './styles.module.css';

export default function GettingStarted() {
    return (
        <section className={cls.section}>
            <div className={cls.container}>
                <div className={cls.sectionHeader}>
                    <h2 className={cls.sectionTitle}>Start Learning in 3 Simple Steps</h2>
                </div>

                <div className={cls.stepsGrid}>
                    <div className={clsx(cls.glassCard, cls.stepCard)}>
                        <div className={cls.stepNum}>01</div>
                        <h3>Choose Your Path</h3>
                        <p>Browse the table of contents and start with the chapter that matches your current level.</p>
                    </div>

                    <div className={clsx(cls.glassCard, cls.stepCard)}>
                        <div className={cls.stepNum}>02</div>
                        <h3>Follow Along</h3>
                        <p>Read the explanations and run the code examples in your own environment as you go.</p>
                    </div>

                    <div className={clsx(cls.glassCard, cls.stepCard)}>
                        <div className={cls.stepNum}>03</div>
                        <h3>Build Projects</h3>
                        <p>Apply what you learn immediately with end-of-chapter projects and challenges.</p>
                    </div>
                </div>

                <div style={{ textAlign: 'center', marginTop: '3rem' }}>
                    <Link className={cls.primaryButton} to="/docs/intro">
                        Start with Chapter 1 →
                    </Link>
                </div>
            </div>
        </section>
    );
}
