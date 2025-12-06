import React, { useState, useEffect } from 'react';
import Link from '@docusaurus/Link';
import { useLocation } from '@docusaurus/router';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import styles from './styles.module.css';

const STORAGE_KEY = 'docusaurus_last_visit';
const DISMISSED_KEY = 'docusaurus_continue_dismissed';

export default function LastVisitBanner(): React.JSX.Element | null {
    const [lastVisit, setLastVisit] = useState<{ path: string; title: string } | null>(null);
    const [visible, setVisible] = useState(false);
    const location = useLocation();

    useEffect(() => {
        if (!ExecutionEnvironment.canUseDOM) return;

        try {
            // Check if dismissed for this session
            const isDismissed = sessionStorage.getItem(DISMISSED_KEY);
            if (isDismissed === 'true') return;

            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const data = JSON.parse(stored);

                // Logic: Show if stored path exists, is different from current, and usually on Homepage
                // Allowing it to show on any page if it's not the stored page, per "Continue Reading" pattern
                const isHomePage = location.pathname === '/' || location.pathname === '/docusurus-frontend/' || location.pathname.endsWith('/intro');

                if (data.path && data.path !== location.pathname && isHomePage) {
                    setLastVisit(data);
                    setVisible(true);
                } else {
                    setVisible(false);
                }
            }
        } catch (e) {
            console.error(e);
        }
    }, [location.pathname]);

    const handleDismiss = () => {
        setVisible(false);
        sessionStorage.setItem(DISMISSED_KEY, 'true');
    };

    if (!visible || !lastVisit) return null;

    return (
        <div className={styles.bannerContainer}>
            <div className={styles.bannerContent}>
                <span className={styles.bannerText}>
                    Welcome back! Continue reading
                    <span className={styles.bannerTitle}> "{lastVisit.title}"</span>
                </span>
                <div className={styles.actions}>
                    <Link to={lastVisit.path} className={styles.continueLink}>
                        Continue
                        <svg
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            className={styles.arrowIcon}
                        >
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </Link>
                    <button
                        onClick={handleDismiss}
                        className={styles.dismissBtn}
                        aria-label="Dismiss"
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
}
