import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

type FontSize = 'default' | 'lg' | 'xl';

const STORAGE_KEY = 'docusaurus_font_size';
const FONT_SIZE_LABELS: Record<FontSize, string> = {
    default: 'A',
    lg: 'A+',
    xl: 'A++',
};

/**
 * Font Size Toggle Component
 * Cycles through font sizes: default → lg → xl → default
 */
export default function FontSizeToggle(): JSX.Element {
    const [fontSize, setFontSize] = useState<FontSize>('default');
    const [mounted, setMounted] = useState(false);

    // Initialize from localStorage
    useEffect(() => {
        setMounted(true);
        if (typeof window === 'undefined') return;

        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            const size = stored ? JSON.parse(stored) : 'default';
            setFontSize(size);
            applyFontSize(size);
        } catch {
            setFontSize('default');
        }
    }, []);

    const applyFontSize = (size: FontSize) => {
        if (typeof document === 'undefined') return;
        if (size === 'default') {
            document.documentElement.removeAttribute('data-font-size');
        } else {
            document.documentElement.setAttribute('data-font-size', size);
        }
    };

    const cycleFontSize = () => {
        const order: FontSize[] = ['default', 'lg', 'xl'];
        const currentIndex = order.indexOf(fontSize);
        const newSize = order[(currentIndex + 1) % order.length];

        setFontSize(newSize);
        applyFontSize(newSize);

        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(newSize));
        } catch {
            // Storage error
        }
    };

    if (!mounted) return <div className={styles.placeholder} />;

    return (
        <button
            className={styles.toggleBtn}
            onClick={cycleFontSize}
            aria-label={`Font size: ${fontSize}. Click to change.`}
            title={`Font size: ${fontSize === 'default' ? 'Normal' : fontSize === 'lg' ? 'Large' : 'Extra Large'}`}
        >
            <span className={styles.label}>{FONT_SIZE_LABELS[fontSize]}</span>
        </button>
    );
}
