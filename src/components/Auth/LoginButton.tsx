import React from "react";
import styles from "./styles.module.css";
import { useAuthUI } from "./AuthUIContext";

/**
 * Login/Signup Button Component
 * Triggers the global login modal
 */
export default function LoginButton() {
    const { openLoginModal } = useAuthUI();

    return (
        <button
            onClick={openLoginModal}
            className={styles.authBtnPrimary}
        >
            Get Started
        </button>
    );
}
