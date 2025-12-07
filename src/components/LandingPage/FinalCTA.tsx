import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import cls from './styles.module.css';

export default function FinalCTA() {
    return (
        <section className={clsx(cls.section, cls.ctaSection)}>
            <div className={cls.container}>
                <div className={cls.ctaContent}>
                    <h2 className={cls.ctaTitle}>Ready to Build Intelligent Robots?</h2>
                    <p className={cls.subheadline}>
                        Join thousands of engineers mastering the future of physical AI.
                    </p>
                    <div style={{ marginTop: '2rem' }}>
                        <Link className={clsx(cls.primaryButton, cls.ctaButton)} to="/docs/intro">
                            Start Reading Now
                        </Link>
                    </div>
                </div>
            </div>
        </section>
    );
}
