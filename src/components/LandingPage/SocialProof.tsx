import React from 'react';
import clsx from 'clsx';
import cls from './styles.module.css';

const testimonials = [
    {
        quote: "The most comprehensive and well-structured documentation I've found. Finally cleared up concepts I've struggled with for months.",
        author: "Sarah Chen",
        role: "Full-Stack Developer",
        avatar: "https://i.pravatar.cc/150?u=sarah"
    },
    {
        quote: "Perfect balance of theory and practice. The code examples are production-quality and actually teach you proper patterns.",
        author: "Marcus Johnson",
        role: "Senior Engineer",
        avatar: "https://i.pravatar.cc/150?u=marcus"
    },
    {
        quote: "This is now my go-to reference. Clear, current, and covers everything from basics to advanced topics.",
        author: "Aisha Patel",
        role: "Software Architect",
        avatar: "https://i.pravatar.cc/150?u=aisha"
    }
];

export default function SocialProof() {
    return (
        <section className={cls.section} style={{ background: 'rgba(0,0,0,0.3)' }}>
            <div className={cls.container}>
                <div className={cls.sectionHeader}>
                    <h2 className={cls.sectionTitle}>Trusted by Developers Worldwide</h2>
                </div>

                <div className={cls.testimonialGrid}>
                    {testimonials.map((t, idx) => (
                        <div key={idx} className={clsx(cls.glassCard, cls.testimonialCard)}>
                            <div className={cls.quoteIcon}>“</div>
                            <p className={cls.quoteText}>{t.quote}</p>
                            <div className={cls.authorInfo}>
                                <img src={t.avatar} alt={t.author} className={cls.authorAvatar} />
                                <div>
                                    <div className={cls.authorName}>{t.author}</div>
                                    <div className={cls.authorRole}>{t.role}</div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                <div className={cls.statsRow}>
                    <div className={cls.statItem}>
                        <div className={cls.statNum}>15.3k</div>
                        <div className={cls.statLabel}>GitHub Stars</div>
                    </div>
                    <div className={cls.statItem}>
                        <div className={cls.statNum}>250k+</div>
                        <div className={cls.statLabel}>Downloads</div>
                    </div>
                    <div className={cls.statItem}>
                        <div className={cls.statNum}>5k+</div>
                        <div className={cls.statLabel}>Community</div>
                    </div>
                </div>
            </div>
        </section>
    );
}
