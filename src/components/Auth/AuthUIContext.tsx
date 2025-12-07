import { createContext, useContext } from 'react';

interface AuthUIContextType {
    isLoginModalOpen: boolean;
    openLoginModal: () => void;
    closeLoginModal: () => void;
}

export const AuthUIContext = createContext<AuthUIContextType | undefined>(undefined);

export function useAuthUI() {
    const context = useContext(AuthUIContext);
    if (context === undefined) {
        throw new Error('useAuthUI must be used within an AuthUIProvider');
    }
    return context;
}
