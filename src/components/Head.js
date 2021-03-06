import React from 'react';
import Helmet from 'react-helmet';
import { StaticQuery, graphql } from 'gatsby';

const Head = ({ title, description }) => {
    const [ isClient, setIsClient ] = React.useState(false);
    React.useEffect(() => {
        setIsClient(true);
    }, false);
    return (
        <StaticQuery
            query={query}
            render={data => {
                const lang = 'en';
                const siteMetadata = data.site.siteMetadata;
                const pageTitle = title ? `${title} · ${siteMetadata.title}` : `${siteMetadata.title}`;
                const pageDesc = description || siteMetadata.description;
                const image = '/social.jpg';
                const meta = [
                    {
                        name: 'description',
                        content: pageDesc
                    },
                    {
                        property: 'og:title',
                        content: pageTitle
                    },
                    {
                        property: 'og:description',
                        content: pageDesc
                    },
                    {
                        property: 'og:type',
                        content: `website`
                    },
                    {
                        property: 'og:site_name',
                        content: siteMetadata.title
                    },
                    {
                        property: 'og:image',
                        content: image
                    },
                    {
                        name: 'twitter:card',
                        content: 'summary_large_image'
                    },
                    {
                        name: 'twitter:image',
                        content: image
                    },
                    {
                        name: 'twitter:creator',
                        content: `@${siteMetadata.twitter}`
                    },
                    {
                        name: 'twitter:site',
                        content: `@${siteMetadata.twitter}`
                    },
                    {
                        name: 'twitter:title',
                        content: pageTitle
                    },
                    {
                        name: 'twitter:description',
                        content: pageDesc
                    }
                ];
                const link = [
                    {
                        rel: 'stylesheet',
                        href: 'https://cdn.jsdelivr.net/npm/@allenai/varnish@0.8.11/dist/theme.min.css'
                    },
                    {
                        rel: 'icon',
                        href: '/favicon.ico',
                        type: 'image/x-icon'
                    },
                    {
                        rel: 'apple-touch-icon',
                        href: '/icons/apple-touch-icon.png'
                    }
                ];

                const gaId = siteMetadata.googleAnalyticsId;

                return (
                    <Helmet
                        defer={false}
                        htmlAttributes={{ lang }}
                        title={pageTitle}
                        meta={meta}
                        link={link}>
                        {gaId !== '' && (
                            <script
                                async=""
                                src={`https://www.googletagmanager.com/gtag/js?id=${gaId}`}
                            />
                        )}
                        {gaId !== '' && (
                            <script>{`
                                window.dataLayer = window.dataLayer || [];
                                function gtag() { dataLayer.push(arguments); }
                                gtag('js', new Date());
                                gtag('config', '${gaId}');
                            `}</script>
                        )}
                        {/* We only trigger this on the client to avoid double-counting:
                          * - The `isClient` gates prevents the script from being includeed in
                          *   the SSR'd version, which causes the double-count bug to occur.
                          * - We intentionally omit the `data-spa` flag, as this component is
                          *   re-rendered with every pageview, thereby triggering a Skiff Stats
                          *   tracking request to fire.
                          * See: https://github.com/allenai/skiff/issues/891 */
                         isClient ? (
                            <script
                                src="https://stats.allenai.org/init.min.js"
                                data-app-name="allennlp-guide"
                                async
                            />
                        ) : null}
                    </Helmet>
                );
            }}
        />
    );
}

export default Head;

const query = graphql`
    query DefaultSEOQuery {
        site {
            siteMetadata {
                title
                description
                twitter
                googleAnalyticsId
            }
        }
    }
`;
