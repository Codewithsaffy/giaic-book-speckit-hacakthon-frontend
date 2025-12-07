import React from 'react';
import clsx from 'clsx';
import cls from './styles.module.css';

const faqs = [
    { q: "Do I need a physical robot to follow along?", a: "No! The digital twin module covers full physics simulation in Gazebo and Isaac Sim, allowing you to learn without hardware." },
    { q: "What prerequisites are needed?", a: "Basic Python knowledge is required. We cover ROS 2 fundamentals from scratch." },
    { q: "Is this updated for the latest ROS 2 version?", a: "Yes, all content targets ROS 2 Jazzy Jalisco (2025)." },
    { q: "Does it cover Reinforcement Learning?", a: "Yes, Module 3 focuses on training policies in Isaac Lab and deploying them to robots." },
    { q: "Is the content available in Urdu?", a: "Absolutley. We provide full bilingual support with Roman Urdu translations for accessible learning." },
];

export default function FAQ() {
    const [openIndex, setOpenIndex] = React.useState<number | null>(0);

    const toggle = (idx: number) => {
        setOpenIndex(openIndex === idx ? null : idx);
    };

    return (
        <section className={cls.section}>
            <div className={cls.container}>
                <div className={cls.sectionHeader}>
                    <h2 className={cls.sectionTitle}>Frequently Asked Questions</h2>
                </div>

                <div className={cls.faqList}>
                    {faqs.map((item, idx) => (
                        <div key={idx} className={clsx(cls.faqItem, openIndex === idx && cls.faqOpen)}>
                            <button className={cls.faqQuestion} onClick={() => toggle(idx)}>
                                <span>{item.q}</span>
                                <span className={cls.faqIcon}>{openIndex === idx ? '−' : '+'}</span>
                            </button>
                            <div className={cls.faqAnswer} style={{ maxHeight: openIndex === idx ? '200px' : '0' }}>
                                <div className={cls.faqAnswerInner}>{item.a}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
