import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

const STORAGE_KEY = 'docusaurus_last_visit';
const DISMISSED_KEY = 'docusaurus_continue_dismissed';

// Naive implementation: just track the last page visited
// We rely on the browser's localStorage
if (ExecutionEnvironment.canUseDOM) {
    // We only want to track actual page visits, not just any route update
    // But client modules run on every navigation

    // We hook into the history listener if possible, or just run top level
    // Docusaurus client modules are imported for side effects

    // A simple way to track is to listen to onRouteDidUpdate
    // But client-modules interface in Docusaurus v3 is different. 
    // We can export an onRouteDidUpdate function.
}

export function onRouteDidUpdate({ location, previousLocation }) {
    if (!ExecutionEnvironment.canUseDOM) return;

    // Don't track if we are on the same page (hash change)
    if (previousLocation && location.pathname === previousLocation.pathname) {
        return;
    }

    // Filter out non-content pages if necessary (like search, or 404 handled by router)
    // For now, track everything that isn't root if we want
    const currentPath = location.pathname;

    // Helper to get title - might be delayed, so we store path mostly
    // We can try to grab document.title
    requestAnimationFrame(() => {
        const title = document.title.split('|')[0].trim(); // Remove site name suffix if present

        const visitData = {
            path: currentPath,
            title: title,
            timestamp: Date.now()
        };

        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(visitData));
            // Reset dismissed state on new page visit so banner can show again if logic dictates
            // Or keep it dismissed for the session. 
            // Generous logic: if they navigate, we might show it again next time they come back to the site specificially?
            // Actually "Continue Reading" usually means "last session". 
            // If I am navigating NOW, I don't need a continue reading banner. 
            // The banner is for when I loose context. 

            // Let's NOT reset dismissed key here, otherwise it pops up every time we navigate back to home?
            // Requirement: "Show ... if current URL != stored URL". 
        } catch (e) {
            console.error('Failed to save visit history', e);
        }
    });
}
