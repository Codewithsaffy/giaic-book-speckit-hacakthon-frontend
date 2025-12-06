import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import styles from './styles.module.css';

interface Bookmark {
    path: string;
    title: string;
}

const STORAGE_KEY = 'docusaurus_bookmarks';

/**
 * Bookmark Button Component
 * Heart icon that toggles bookmark state for current page
 */
export default function BookmarkButton(): React.JSX.Element {
    const location = useLocation();
    const [isBookmarked, setIsBookmarked] = useState(false);
    const [mounted, setMounted] = useState(false);

    // Get current page title
    const getPageTitle = (): string => {
        if (typeof document === 'undefined') return '';
        const h1 = document.querySelector('article h1, .markdown h1, h1');
        if (h1) return h1.textContent?.trim() || '';
        return document.title.split('|')[0]?.trim() || 'Page';
    };

    // Check if current page is bookmarked
    useEffect(() => {
        setMounted(true);
        if (typeof window === 'undefined') return;

        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            const bookmarks: Bookmark[] = stored ? JSON.parse(stored) : [];
            setIsBookmarked(bookmarks.some(b => b.path === location.pathname));
        } catch {
            setIsBookmarked(false);
        }
    }, [location.pathname]);

    const toggleBookmark = () => {
        if (typeof window === 'undefined') return;

        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            let bookmarks: Bookmark[] = stored ? JSON.parse(stored) : [];

            if (isBookmarked) {
                bookmarks = bookmarks.filter(b => b.path !== location.pathname);
            } else {
                const title = getPageTitle();
                bookmarks.push({ path: location.pathname, title });
            }

            localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks));
            setIsBookmarked(!isBookmarked);

            // Dispatch event for other components
            window.dispatchEvent(new CustomEvent('bookmarks-updated'));
        } catch {
            // Storage error
        }
    };

    if (!mounted) return <div className={styles.placeholder} />;

    return (
        <button
            className={`${styles.bookmarkBtn} ${isBookmarked ? styles.active : ''}`}
            onClick={toggleBookmark}
            aria-label={isBookmarked ? 'Remove bookmark' : 'Add bookmark'}
            title={isBookmarked ? 'Remove from bookmarks' : 'Add to bookmarks'}
        >
            <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill={isBookmarked ? 'currentColor' : 'none'}
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
            >
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
            </svg>
        </button>
    );
}
