/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        // Proxy all /api/* requests to the FastAPI backend
        source: '/api/:path*',
        destination: 'http://localhost:8001/api/:path*',
      },
      {
        // Proxy the health-check endpoint used by the status indicator
        source: '/health',
        destination: 'http://localhost:8001/health',
      },
    ]
  },
}

module.exports = nextConfig
