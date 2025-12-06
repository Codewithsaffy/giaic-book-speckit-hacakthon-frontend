import React from 'react';
import Content from '@theme-original/DocItem/Content';
import type ContentType from '@theme/DocItem/Content';
import type { WrapperProps } from '@docusaurus/types';
import BookmarkButton from '@site/src/components/BookmarkButton';
import styles from './styles.module.css';

type Props = WrapperProps<typeof ContentType>;

/**
 * DocItem/Content wrapper that injects bookmark button at the top
 */
export default function ContentWrapper(props: Props): React.JSX.Element {
    return (
        <div className={styles.contentWrapper}>
            <div className={styles.bookmarkContainer}>
                <BookmarkButton />
            </div>
            <Content {...props} />
        </div>
    );
}
