import React from "react";
import { useSession } from "@site/src/lib/auth-client";
import LoginButton from "@site/src/components/Auth/LoginButton";
import UserMenu from "@site/src/components/Auth/UserMenu";

/**
 * Navbar Auth Component
 * Shows LoginButton or UserMenu based on session state
 */
export default function NavbarAuth() {
    const { data: session, isPending } = useSession();

    if (isPending) {
        return <div style={{ padding: "0 1rem" }}>Loading...</div>;
    }

    return <div>{session ? <UserMenu /> : <LoginButton />}</div>;
}
