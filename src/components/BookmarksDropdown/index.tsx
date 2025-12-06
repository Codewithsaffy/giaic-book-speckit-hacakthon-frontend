import React, { useState, useEffect, useRef } from 'react';
import styles from './styles.module.css';

interface Bookmark {
    path: string;
    title: string;
}

const STORAGE_KEY = 'docusaurus_bookmarks';

/**
 * Bookmarks Dropdown Component
 * Navbar dropdown showing all bookmarked pages
 */
export default function BookmarksDropdown(): React.JSX.Element {
    const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const [mounted, setMounted] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Load bookmarks on mount
    useEffect(() => {
        setMounted(true);
        loadBookmarks();

        // Listen for bookmark updates
        const handleUpdate = () => loadBookmarks();
        window.addEventListener('bookmarks-updated', handleUpdate);
        window.addEventListener('storage', handleUpdate);

        return () => {
            window.removeEventListener('bookmarks-updated', handleUpdate);
            window.removeEventListener('storage', handleUpdate);
        };
    }, []);

    // Close dropdown on outside click
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
                setIsOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const loadBookmarks = () => {
        if (typeof window === 'undefined') return;
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            setBookmarks(stored ? JSON.parse(stored) : []);
        } catch {
            setBookmarks([]);
        }
    };

    const removeBookmark = (path: string, e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();

        if (typeof window === 'undefined') return;
        try {
            const updated = bookmarks.filter(b => b.path !== path);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
            setBookmarks(updated);
            window.dispatchEvent(new CustomEvent('bookmarks-updated'));
        } catch {
            // Storage error
        }
    };

    if (!mounted) return <div className={styles.placeholder} />;

    return (
        <div className={styles.dropdown} ref={dropdownRef}>
            <button
                className={styles.dropdownBtn}
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Bookmarks"
                title="Your bookmarks"
            >
                <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                >
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                </svg>
                {bookmarks.length > 0 && (
                    <span className={styles.badge}>{bookmarks.length}</span>
                )}
            </button>

            {isOpen && (
                <div className={styles.menu}>
                    <div className={styles.menuHeader}>
                        <span>Bookmarks</span>
                        <span className={styles.count}>{bookmarks.length}</span>
                    </div>

                    {bookmarks.length === 0 ? (
                        <div className={styles.emptyState}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                            </svg>
                            <p>No bookmarks yet</p>
                            <span>Click the heart icon on any page to save it here</span>
                        </div>
                    ) : (
                        <ul className={styles.menuList}>
                            {bookmarks.map((bookmark) => (
                                <li key={bookmark.path} className={styles.menuItem}>
                                    <a href={bookmark.path} className={styles.menuLink}>
                                        <span className={styles.linkTitle}>{bookmark.title}</span>
                                        <span className={styles.linkPath}>{bookmark.path}</span>
                                    </a>
                                    <button
                                        className={styles.removeBtn}
                                        onClick={(e) => removeBookmark(bookmark.path, e)}
                                        aria-label="Remove bookmark"
                                    >
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <line x1="18" y1="6" x2="6" y2="18" />
                                            <line x1="6" y1="6" x2="18" y2="18" />
                                        </svg>
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            )}
        </div>
    );
}
