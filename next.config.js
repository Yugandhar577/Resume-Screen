/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: process.env.BUILD_STANDALONE === 'true' ? 'standalone' : undefined,
  async rewrites() {
    const backendUrl = (process.env.BACKEND_URL || 'http://127.0.0.1:8001').replace(/\/$/, '')
    return [
      {
        // Proxy all /api/* requests to the FastAPI backend
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      {
        // Proxy the health-check endpoint used by the status indicator
        source: '/health',
        destination: `${backendUrl}/health`,
      },
    ]
  },
}

module.exports = nextConfig
