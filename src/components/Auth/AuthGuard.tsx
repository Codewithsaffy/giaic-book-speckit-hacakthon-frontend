import React from 'react';
import { useSession } from "@site/src/lib/auth-client";
import { useAuthUI } from './AuthUIContext';

interface AuthGuardProps {
    children: React.ReactNode;
}

/**
 * Wraps a component and intercepts clicks to enforce authentication.
 * If not authenticated, opens the login modal instead of allowing the interaction.
 */
export default function AuthGuard({ children }: AuthGuardProps) {
    const { data: session } = useSession();
    const { openLoginModal } = useAuthUI();

    const handleCapture = (e: React.MouseEvent) => {
        if (!session) {
            e.preventDefault();
            e.stopPropagation();
            openLoginModal();
        }
    };

    if (!session) {
        return (
            <div onClickCapture={handleCapture} style={{ display: 'contents' }}>
                {children}
            </div>
        );
    }

    return <>{children}</>;
}
