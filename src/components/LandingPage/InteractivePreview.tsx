import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function InteractivePreview() {
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                <div className={styles.sectionHeader}>
                    <h2 className={styles.sectionTitle}>Real World Code. Real World Physics.</h2>
                    <p className={styles.subheadline}>
                        Connect AI brains to physical bodies with production-ready ROS 2 and Isaac Sim integrations.
                    </p>
                </div>

                <div className={styles.previewWindow}>
                    <div className={styles.previewHeader}>
                        <div className={styles.previewControls}>
                            <span className={clsx(styles.controlDot, styles.redDot)}></span>
                            <span className={clsx(styles.controlDot, styles.yellowDot)}></span>
                            <span className={clsx(styles.controlDot, styles.greenDot)}></span>
                        </div>
                        <div className={styles.previewTabs}>
                            <div className={clsx(styles.previewTab, styles.activeTab)}>robot_controller.py</div>
                            <div className={styles.previewTab}>robot.urdf</div>
                        </div>
                    </div>
                    <div className={styles.previewBody}>
                        <div className={styles.codePane}>
                            <pre><code>
                                <span className={styles.kw}>import</span> <span className={styles.fn}>rclpy</span><br />
                                <span className={styles.kw}>from</span> <span className={styles.fn}>rclpy.node</span> <span className={styles.kw}>import</span> Node<br />
                                <span className={styles.kw}>from</span> <span className={styles.fn}>geometry_msgs.msg</span> <span className={styles.kw}>import</span> Twist<br />
                                <br />
                                <span className={styles.kw}>class</span> <span className={styles.cmp}>RobotController</span>(Node):<br />
                                &nbsp;&nbsp;<span className={styles.kw}>def</span> <span className={styles.fn}>__init__</span>(self):<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;<span className={styles.fn}>super</span>().__init__(<span className={styles.str}>'robot_controller'</span>)<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;self.publisher_ = self.<span className={styles.fn}>create_publisher</span>(Twist, <span className={styles.str}>'/cmd_vel'</span>, 10)<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;self.timer = self.<span className={styles.fn}>create_timer</span>(0.5, self.timer_callback)<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;self.<span className={styles.fn}>get_logger</span>().<span className={styles.fn}>info</span>(<span className={styles.str}>"Controller Node Started 🚀"</span>)<br />
                                <br />
                                &nbsp;&nbsp;<span className={styles.kw}>def</span> <span className={styles.fn}>timer_callback</span>(self):<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;msg = <span className={styles.cmp}>Twist</span>()<br />
                                &nbsp;&nbsp;&nbsp;&nbsp;msg.linear.x = <span className={styles.bool}>2.0</span><br />
                                &nbsp;&nbsp;&nbsp;&nbsp;msg.angular.z = <span className={styles.bool}>0.5</span><br />
                                &nbsp;&nbsp;&nbsp;&nbsp;self.publisher_.<span className={styles.fn}>publish</span>(msg)<br />
                            </code></pre>
                        </div>
                        <div className={styles.previewPane}>
                            <div className={styles.mockContent}>
                                <div className={styles.mockSpinner}></div>
                                <div className={styles.mockText} style={{ fontFamily: 'monospace', fontSize: '0.9rem', textAlign: 'left' }}>
                                    [INFO] [1715629.213]: Controller Node Started 🚀<br />
                                    [INFO] [1715629.713]: Publishing Cmd: Linear: 2.0, Angular: 0.5<br />
                                    [INFO] [1715630.213]: Publishing Cmd: Linear: 2.0, Angular: 0.5<br />
                                    <span style={{ color: '#27c93f' }}>&gt; _</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                    <Link className={styles.primaryButton} to="/docs/intro">
                        Build Your First Node →
                    </Link>
                </div>
            </div>
        </section>
    );
}
