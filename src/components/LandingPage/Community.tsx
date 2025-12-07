import React from 'react';
import Link from '@docusaurus/Link';
import cls from './styles.module.css';

export default function Community() {
    return (
        <section className={cls.section} style={{ background: 'linear-gradient(180deg, rgba(20,20,23,0) 0%, rgba(20,20,23,0.8) 100%)' }}>
            <div className={cls.container}>
                <div className={cls.communityBox}>
                    <h2 className={cls.sectionTitle}>Join the Community</h2>
                    <p className={cls.subheadline}>
                        Ask questions, share projects, and learn with 5,000+ developers.
                    </p>

                    <div className={cls.communityGrid}>
                        <Link to="https://discord.gg/example" className={cls.communityCard}>
                            <div className={cls.communityIcon}>💬</div>
                            <h3>Discussion Forum</h3>
                            <p>Join the conversation</p>
                        </Link>
                        <Link to="https://github.com/example/repo" className={cls.communityCard}>
                            <div className={cls.communityIcon}>🐛</div>
                            <h3>Report Issues</h3>
                            <p>Contribute on GitHub</p>
                        </Link>
                        <div className={cls.communityCard}>
                            <div className={cls.communityIcon}>📫</div>
                            <h3>Newsletter</h3>
                            <p>Get monthly updates</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
