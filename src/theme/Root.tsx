import React from "react";
import AuthProvider from "@site/src/components/Auth/AuthProvider";
import AuthUIProvider from "@site/src/components/Auth/AuthUIProvider";

import AskButton from '@site/src/components/AskButton';

/**
 * Root component wrapper for Docusaurus
 * Wraps the entire app with AuthProvider
 */
export default function Root({ children }: { children: React.ReactNode }) {
    return (
        <AuthProvider>
            <AuthUIProvider>
                {children}
                <AskButton />
            </AuthUIProvider>
        </AuthProvider>
    );
}
