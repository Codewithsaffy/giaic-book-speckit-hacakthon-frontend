import React, { useState } from 'react';
import ChatWidget from '../ChatWidget';

export default function AskButton(): JSX.Element {
    const [isChatOpen, setIsChatOpen] = useState(false);

    return (
        <>
            {!isChatOpen && (
                <div className="ask-button-container" onClick={() => setIsChatOpen(!isChatOpen)}>
                    <span className="ask-button-text">Ask a question...</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="ask-button-shortcut">Ctrl+I</span>
                        <span className="ask-button-arrow">↑</span>
                    </div>
                </div>
            )}
            {isChatOpen && <ChatWidget onClose={() => setIsChatOpen(false)} />}
        </>
    );
}
