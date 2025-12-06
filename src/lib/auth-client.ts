import { createAuthClient } from "better-auth/react";


export const authClient = createAuthClient({
    baseURL: process.env.NODE_ENV === "production"
        ? "https://roboticai-auth.vercel.app"
        : "http://localhost:3000",
});
// Export commonly used hooks and methods
export const { useSession, signIn, signUp, signOut } = authClient;
