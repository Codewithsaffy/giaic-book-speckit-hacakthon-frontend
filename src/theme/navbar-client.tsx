import React from 'react';
import { createRoot } from 'react-dom/client';
import NavbarAuth from '@site/src/components/Auth/NavbarAuth';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

if (ExecutionEnvironment.canUseDOM) {
    // Mount NavbarAuth component
    const authRoot = document.getElementById('navbar-auth-root');
    if (authRoot) {
        const root = createRoot(authRoot);
        root.render(<NavbarAuth />);
    }

    // Handle custom theme toggle
    const themeToggle = document.getElementById('theme-toggle-custom');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
}
