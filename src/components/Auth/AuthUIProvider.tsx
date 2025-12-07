import React, { useState } from 'react';
import { AuthUIContext } from './AuthUIContext';
import LoginModal from './LoginModal';

export default function AuthUIProvider({ children }: { children: React.ReactNode }) {
    const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);

    const openLoginModal = () => setIsLoginModalOpen(true);
    const closeLoginModal = () => setIsLoginModalOpen(false);

    return (
        <AuthUIContext.Provider value={{ isLoginModalOpen, openLoginModal, closeLoginModal }}>
            {children}
            <LoginModal isOpen={isLoginModalOpen} onClose={closeLoginModal} />
        </AuthUIContext.Provider>
    );
}
