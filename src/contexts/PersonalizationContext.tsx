import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

/**
 * Bookmark item structure
 */
interface Bookmark {
    path: string;
    title: string;
}

/**
 * Font size options
 */
type FontSize = 'default' | 'lg' | 'xl';

/**
 * Personalization context state
 */
interface PersonalizationContextType {
    // Bookmarks
    bookmarks: Bookmark[];
    addBookmark: (path: string, title: string) => void;
    removeBookmark: (path: string) => void;
    isBookmarked: (path: string) => boolean;
    // Font size
    fontSize: FontSize;
    cycleFontSize: () => void;
    // Continue reading
    lastVisitedPath: string | null;
    lastVisitedTitle: string | null;
    dismissContinueReading: () => void;
    continueReadingDismissed: boolean;
}

const PersonalizationContext = createContext<PersonalizationContextType | null>(null);

const STORAGE_KEYS = {
    BOOKMARKS: 'docusaurus_bookmarks',
    FONT_SIZE: 'docusaurus_font_size',
    LAST_PATH: 'docusaurus_last_path',
    LAST_TITLE: 'docusaurus_last_title',
    CONTINUE_DISMISSED: 'docusaurus_continue_dismissed',
};

/**
 * Safe localStorage getter (SSR-safe)
 */
function getStorageItem<T>(key: string, defaultValue: T): T {
    if (typeof window === 'undefined') return defaultValue;
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch {
        return defaultValue;
    }
}

/**
 * Safe localStorage setter (SSR-safe)
 */
function setStorageItem<T>(key: string, value: T): void {
    if (typeof window === 'undefined') return;
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch {
        // Storage full or disabled
    }
}

/**
 * PersonalizationProvider component
 * Provides bookmarks, font size, and continue reading state
 */
export function PersonalizationProvider({ children }: { children: ReactNode }) {
    const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
    const [fontSize, setFontSize] = useState<FontSize>('default');
    const [lastVisitedPath, setLastVisitedPath] = useState<string | null>(null);
    const [lastVisitedTitle, setLastVisitedTitle] = useState<string | null>(null);
    const [continueReadingDismissed, setContinueReadingDismissed] = useState(false);
    const [mounted, setMounted] = useState(false);

    // Initialize state from localStorage on mount
    useEffect(() => {
        setBookmarks(getStorageItem(STORAGE_KEYS.BOOKMARKS, []));
        setFontSize(getStorageItem(STORAGE_KEYS.FONT_SIZE, 'default'));
        setLastVisitedPath(getStorageItem(STORAGE_KEYS.LAST_PATH, null));
        setLastVisitedTitle(getStorageItem(STORAGE_KEYS.LAST_TITLE, null));
        setContinueReadingDismissed(getStorageItem(STORAGE_KEYS.CONTINUE_DISMISSED, false));
        setMounted(true);
    }, []);

    // Apply font size to HTML element
    useEffect(() => {
        if (!mounted) return;
        if (fontSize === 'default') {
            document.documentElement.removeAttribute('data-font-size');
        } else {
            document.documentElement.setAttribute('data-font-size', fontSize);
        }
        setStorageItem(STORAGE_KEYS.FONT_SIZE, fontSize);
    }, [fontSize, mounted]);

    // Persist bookmarks
    useEffect(() => {
        if (!mounted) return;
        setStorageItem(STORAGE_KEYS.BOOKMARKS, bookmarks);
    }, [bookmarks, mounted]);

    const addBookmark = useCallback((path: string, title: string) => {
        setBookmarks(prev => {
            if (prev.some(b => b.path === path)) return prev;
            return [...prev, { path, title }];
        });
    }, []);

    const removeBookmark = useCallback((path: string) => {
        setBookmarks(prev => prev.filter(b => b.path !== path));
    }, []);

    const isBookmarked = useCallback((path: string) => {
        return bookmarks.some(b => b.path === path);
    }, [bookmarks]);

    const cycleFontSize = useCallback(() => {
        setFontSize(prev => {
            const order: FontSize[] = ['default', 'lg', 'xl'];
            const currentIndex = order.indexOf(prev);
            return order[(currentIndex + 1) % order.length];
        });
    }, []);

    const dismissContinueReading = useCallback(() => {
        setContinueReadingDismissed(true);
        setStorageItem(STORAGE_KEYS.CONTINUE_DISMISSED, true);
    }, []);

    const value: PersonalizationContextType = {
        bookmarks,
        addBookmark,
        removeBookmark,
        isBookmarked,
        fontSize,
        cycleFontSize,
        lastVisitedPath,
        lastVisitedTitle,
        dismissContinueReading,
        continueReadingDismissed,
    };

    return (
        <PersonalizationContext.Provider value={value}>
            {children}
        </PersonalizationContext.Provider>
    );
}

/**
 * Hook to access personalization context
 */
export function usePersonalization(): PersonalizationContextType {
    const context = useContext(PersonalizationContext);
    if (!context) {
        throw new Error('usePersonalization must be used within PersonalizationProvider');
    }
    return context;
}

export default PersonalizationContext;
