import React, { type ReactNode } from 'react';
import { useThemeConfig } from '@docusaurus/theme-common';
import { useNavbarMobileSidebar } from '@docusaurus/theme-common/internal';
import NavbarColorModeToggle from '@theme/Navbar/ColorModeToggle';
import NavbarMobileSidebarToggle from '@theme/Navbar/MobileSidebar/Toggle';
import NavbarLogo from '@theme/Navbar/Logo';
import NavbarAuth from '@site/src/components/Auth/NavbarAuth';
import LanguageToggle from '@site/src/components/LanguageToggle';
import BookmarksDropdown from '@site/src/components/BookmarksDropdown';
import FontSizeToggle from '@site/src/components/FontSizeToggle';
import SearchBar from '@theme/SearchBar';
import AuthGuard from '@site/src/components/Auth/AuthGuard';
import styles from './styles.module.css';

export default function NavbarContent(): ReactNode {
  const mobileSidebar = useNavbarMobileSidebar();
  const { navbar } = useThemeConfig();

  return (
    <div className={styles.navbarContainer}>
      {/* Left Section: Mobile Toggle + Logo + Title + Language */}
      <div className={styles.navbarLeft}>
        {!mobileSidebar.disabled && <NavbarMobileSidebarToggle />}
        <NavbarLogo />

        {/* Language Toggle Button */}
        <AuthGuard>
          <LanguageToggle />
        </AuthGuard>
      </div>

      {/* Center Section: Search Bar */}
      <div className={styles.navbarCenter}>
        <div
          className={styles.searchBar}
          onClick={(e) => {
            const container = e.currentTarget;
            const input = container.querySelector('input');
            if (input) input.focus();
          }}
        >
          <SearchBar />
        </div>
      </div>

      {/* Right Section: Bookmarks + Font Size + Auth + Theme Toggle */}
      <div className={styles.navbarRight}>
        <AuthGuard>
          <BookmarksDropdown />
          <FontSizeToggle />
        </AuthGuard>

        <NavbarAuth />
        <NavbarColorModeToggle className={styles.colorModeToggle} />
      </div>
    </div>
  );
}
