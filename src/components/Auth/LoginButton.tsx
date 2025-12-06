import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { signIn, signUp } from "@site/src/lib/auth-client";
import styles from "./styles.module.css";

/**
 * Login/Signup Button Component
 * Shows a modal for authentication
 */
export default function LoginButton() {
    const [isOpen, setIsOpen] = useState(false);
    const [isSignUp, setIsSignUp] = useState(false);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [name, setName] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        return () => setMounted(false);
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError("");

        try {
            if (isSignUp) {
                await signUp.email({
                    email,
                    password,
                    name,
                });
            } else {
                await signIn.email({
                    email,
                    password,
                    name,
                });
            }
            setIsOpen(false);
            // Reload to update session state
            window.location.reload();
        } catch (err: any) {
            setError(err.message || "Authentication failed");
        } finally {
            setLoading(false);
        }
    };

    const modalContent = isOpen ? (
        <div
            className={styles.modalOverlay}
            onClick={() => setIsOpen(false)}
        >
            <div
                className={styles.modalContent}
                onClick={(e) => e.stopPropagation()}
            >
                <h2 className={styles.modalTitle}>{isSignUp ? "Sign Up" : "Login"}</h2>
                <form onSubmit={handleSubmit}>
                    {isSignUp && (
                        <div className={styles.formGroup}>
                            <label className={styles.label}>Name</label>
                            <input
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                required
                                className={styles.input}
                                placeholder="Enter your name"
                            />
                        </div>
                    )}
                    <div className={styles.formGroup}>
                        <label className={styles.label}>Email</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className={styles.input}
                            placeholder="name@example.com"
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label className={styles.label}>Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className={styles.input}
                            placeholder="••••••••"
                        />
                    </div>
                    {error && (
                        <div className={styles.errorMessage}>
                            {error}
                        </div>
                    )}
                    <button
                        type="submit"
                        disabled={loading}
                        className={styles.submitBtn}
                    >
                        {loading ? "Loading..." : isSignUp ? "Sign Up" : "Login"}
                    </button>
                    <button
                        type="button"
                        onClick={() => setIsSignUp(!isSignUp)}
                        className={styles.switchBtn}
                    >
                        {isSignUp
                            ? "Already have an account? Login"
                            : "Need an account? Sign Up"}
                    </button>
                </form>
            </div>
        </div>
    ) : null;

    return (
        <>
            <button
                onClick={() => setIsOpen(true)}
                className={styles.authBtnPrimary}
            >
                Get Started
            </button>
            {mounted && createPortal(modalContent, document.body)}
        </>
    );
}
