import React from 'react';
import Header from '@theme-original/DocItem/Header';
import type HeaderType from '@theme/DocItem/Header';
import type { WrapperProps } from '@docusaurus/types';
import BookmarkButton from '@site/src/components/BookmarkButton';
import styles from './styles.module.css';

type Props = WrapperProps<typeof HeaderType>;

/**
 * DocItem/Header wrapper that adds a bookmark button next to the title
 */
export default function HeaderWrapper(props: Props): React.JSX.Element {
    return (
        <div className={styles.headerContainer}>
            <Header {...props} />
            <BookmarkButton />
        </div>
    );
}
