import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const stats = [
    {
        label: 'Modular Learning', value: '4', sub: 'Core Modules', icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.27 6.96 12 12.01 20.73 6.96" /><line x1="12" y1="22.08" x2="12" y2="12" /></svg>
        )
    },
    {
        label: 'In-Depth Topics', value: '150+', sub: 'Technical Topics', icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" /></svg>
        )
    },
    {
        label: 'Available In', value: '2', sub: 'Languages', icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="2" y1="12" x2="22" y2="12" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" /></svg>
        )
    },
    {
        label: 'Latest Release', value: '2025', sub: 'Jazzy Edition', icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
        )
    },
];

export default function QuickValue() {
    return (
        <section className={styles.valueSection}>
            <div className={styles.container}>
                <div className={styles.valueGrid}>
                    {stats.map((stat, idx) => (
                        <div key={idx} className={styles.valueCard}>
                            <div className={styles.valueIcon}>{stat.icon}</div>
                            <div className={styles.valueContent}>
                                <div className={styles.valueBig}>{stat.value}</div>
                                <div className={styles.valueSub}>{stat.sub}</div>
                                <div className={styles.valueLabel}>{stat.label}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
