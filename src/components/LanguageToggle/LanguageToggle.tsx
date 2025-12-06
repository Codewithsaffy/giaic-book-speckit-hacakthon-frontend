import React from 'react';
import { useLocation } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './LanguageToggle.module.css';

/**
 * Language Toggle Button Component
 * Switches between English and Roman Urdu locales
 */
export default function LanguageToggle(): React.JSX.Element {
    const { i18n } = useDocusaurusContext();
    const location = useLocation();
    const currentLocale = i18n.currentLocale;
    const isRomanUrdu = currentLocale === 'ur-Latn';

    const toggleLocale = () => {
        const pathWithoutLocale = location.pathname.replace(/^\/ur-Latn/, '');
        const newPath = isRomanUrdu
            ? pathWithoutLocale || '/'
            : `/ur-Latn${location.pathname}`;

        window.location.href = newPath;
    };

    return (
        <button
            onClick={toggleLocale}
            className={styles.toggleBtn}
            aria-label={isRomanUrdu ? 'Switch to English' : 'Switch to Roman Urdu'}
            title={isRomanUrdu ? 'Switch to English' : 'Roman Urdu mein dekhein'}
        >
            <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={styles.globeIcon}
            >
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
            <span className={styles.label}>
                {isRomanUrdu ? 'EN' : 'اردو'}
            </span>
        </button>
    );
}
