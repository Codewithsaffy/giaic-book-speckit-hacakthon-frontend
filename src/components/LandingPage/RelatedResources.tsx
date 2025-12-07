import React from 'react';
import cls from './styles.module.css';

const resources = [
    { title: "Video Tutorial Series", desc: "Watch hands-on coding sessions." },
    { title: "Interactive Challenges", desc: "Test your knowledge with exercises." },
    { title: "Boilerplate Templates", desc: "Start new projects faster." },
    { title: "Cheat Sheets", desc: "Quick reference guides." },
];

export default function RelatedResources() {
    return (
        <section className={cls.section}>
            <div className={cls.container}>
                <div className={cls.sectionHeader}>
                    <h2 className={cls.sectionTitle}>Extend Your Learning</h2>
                </div>

                <div className={cls.resourceGrid}>
                    {resources.map((r, i) => (
                        <div key={i} className={cls.resourceCard}>
                            <div className={cls.resourceIcon}>📚</div>
                            <div>
                                <h4 className={cls.resourceTitle}>{r.title}</h4>
                                <p className={cls.resourceDesc}>{r.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
