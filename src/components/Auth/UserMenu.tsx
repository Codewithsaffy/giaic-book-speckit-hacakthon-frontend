import React, { useState, useRef, useEffect } from "react";
import { useSession, signOut } from "@site/src/lib/auth-client";
import styles from "./styles.module.css";

/**
 * User Menu Component
 * Shows user info and logout button when authenticated
 */
export default function UserMenu() {
    const { data: session, isPending } = useSession();
    const [isOpen, setIsOpen] = useState(false);
    const menuRef = useRef<HTMLDivElement>(null);

    const handleSignOut = async () => {
        await signOut();
        window.location.reload();
    };

    // Close menu when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    if (isPending) {
        return <div className={styles.authBtn}>Loading...</div>;
    }

    if (!session) {
        return null;
    }

    return (
        <div style={{ position: "relative" }} ref={menuRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={styles.userBtn}
            >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
                {session.user?.name || "User"}
            </button>

            {isOpen && (
                <div className={styles.userDropdown}>
                    <div className={styles.userInfo}>
                        <span className={styles.userName}>{session.user?.name || "User"}</span>
                        <span className={styles.userEmail}>{session.user?.email}</span>
                    </div>
                    <button onClick={handleSignOut} className={styles.logoutBtn}>
                        Logout
                    </button>
                </div>
            )}
        </div>
    );
}
