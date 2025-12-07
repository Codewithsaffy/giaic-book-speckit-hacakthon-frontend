import React, { useState } from 'react';
import { useLocation } from '@docusaurus/router';
import { useSession } from "@site/src/lib/auth-client";
import { useAuthUI } from '../Auth/AuthUIContext';
import ChatWidget from '../ChatWidget';

export default function AskButton(): JSX.Element {
    const [isChatOpen, setIsChatOpen] = useState(false);
    const location = useLocation();
    const { data: session } = useSession();
    const { openLoginModal } = useAuthUI();

    // Only show on docs pages (paths starting with /docs)
    // Adjust logic if docs are served from root or different base
    if (!location.pathname.startsWith('/docs')) {
        return null;
    }

    const handleClick = () => {
        if (!session) {
            openLoginModal();
        } else {
            setIsChatOpen(!isChatOpen);
        }
    };

    return (
        <>
            {!isChatOpen && (
                <div className="ask-button-container" onClick={handleClick}>
                    <span className="ask-button-text">Ask a question...</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="ask-button-shortcut">Ctrl+I</span>
                        <span className="ask-button-arrow">↑</span>
                    </div>
                </div>
            )}
            {isChatOpen && session && <ChatWidget onClose={() => setIsChatOpen(false)} />}
        </>
    );
}
