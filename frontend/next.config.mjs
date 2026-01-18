/** @type {import('next').NextConfig} */
const nextConfig = {
    // CRITICAL: Static export for Capacitor - no Node.js server on mobile
    output: 'export',

    // CRITICAL: Disable image optimization (requires server)
    images: {
        unoptimized: true,
    },

    // Output to 'out' directory for Capacitor
    distDir: 'out',

    // Trailing slashes for static file serving
    trailingSlash: true,

    // Disable server actions (not supported in static export)
    experimental: {
        // Required for static export compatibility
    },
};

export default nextConfig;
